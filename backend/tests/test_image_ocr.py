import io
import shutil
import subprocess

import pytest
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import func, select

from app.config import data_path
from app.models import Document, Figure, ImageText, Passage
from app.services.ocr import OCRResult, extract_image_text


def image_bytes(size=(800, 250), text=None):
    image = Image.new('RGB', size, 'white')
    if text:
        ImageDraw.Draw(image).multiline_text((35,35), text, font=ImageFont.load_default(size=40), fill='black', spacing=15)
    output = io.BytesIO(); image.save(output, format='PNG')
    return output.getvalue()


def test_upload_indexes_ocr_preserves_resolution_and_is_idempotent(client, db, monkeypatch):
    calls = []
    def recognize(image):
        calls.append(image.size)
        return OCRResult('Quantum retrieval image text 4271', 'extracted')
    monkeypatch.setattr('app.services.image_text.extract_image_text', recognize)
    r = client.post('/api/v1/documents/upload-image', files={'file':('scan.png',image_bytes((2400,400)))})
    assert r.status_code == 200, r.text
    result = r.json(); did = result['document_id']; fid = result['ocr']['figure_id']
    assert result['passages_created'] == 1
    assert result['ocr']['text'] == 'Quantum retrieval image text 4271'
    assert calls == [(2400,400)]
    source = client.get(f'/api/v1/documents/{did}/source').content
    assert Image.open(io.BytesIO(source)).size == (2400,400)
    for endpoint in ('/api/v1/search', '/api/v1/search/keyword', '/api/v1/search/hybrid'):
        response = client.get(endpoint, params={'q':'quantum retrieval'})
        assert response.status_code == 200, response.text
        assert response.json()['results'][0]['content'] == result['ocr']['text']
    assert client.get(f'/api/v1/documents/{did}/ocr').json()['results'][0]['text'] == result['ocr']['text']
    assert client.post(f'/api/v1/figures/{fid}/ocr?retry=true').json()['status'] == 'extracted'
    assert len(calls) == 1
    assert db.scalar(select(func.count()).select_from(Passage)) == 1


def test_missing_ocr_retains_visual_search_and_can_retry(client, db, monkeypatch):
    monkeypatch.setattr('app.services.image_text.extract_image_text', lambda image: OCRResult('', 'unavailable', 'Install Tesseract'))
    result = client.post('/api/v1/documents/upload-image', files={'file':('scan.png',image_bytes())}).json()
    assert result['ocr']['status'] == 'unavailable' and result['passages_created'] == 0
    fid = result['ocr']['figure_id']
    assert client.get('/api/v1/search/images?q=diagram').json()['results']
    monkeypatch.setattr('app.services.image_text.extract_image_text', lambda image: OCRResult('Recovered text', 'extracted'))
    assert client.post(f'/api/v1/figures/{fid}/ocr').json()['status'] == 'unavailable'
    assert client.post(f'/api/v1/figures/{fid}/ocr?retry=true').json()['text'] == 'Recovered text'
    assert db.scalar(select(func.count()).select_from(Passage)) == 1


def test_ocr_embedding_failure_rolls_back_image_and_files(client, db, monkeypatch):
    monkeypatch.setattr('app.services.image_text.extract_image_text', lambda image: OCRResult('Recognized text', 'extracted'))
    def fail(_):
        raise RuntimeError('Embedding failed')
    monkeypatch.setattr('app.api.workspace.embed_texts', fail)
    with pytest.raises(RuntimeError, match='Embedding failed'):
        client.post('/api/v1/documents/upload-image', files={'file':('scan.png',image_bytes())})
    for model in (Document, Figure, Passage, ImageText):
        assert db.scalar(select(func.count()).select_from(model)) == 0
    assert not list(data_path('sources').rglob('*.png'))
    assert not list(data_path('figures').rglob('*.png'))


def test_ocr_errors_and_sanitization(monkeypatch):
    image = Image.new('RGB',(100,100),'white')
    def missing(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr('app.services.ocr.subprocess.run', missing)
    assert extract_image_text(image).status == 'unavailable'
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('tesseract',60)
    monkeypatch.setattr('app.services.ocr.subprocess.run', timeout)
    assert extract_image_text(image).status == 'timeout'
    monkeypatch.setattr('app.services.ocr.subprocess.run', lambda *args,**kwargs: subprocess.CompletedProcess([],0,b'\x00  \n',b''))
    assert extract_image_text(image).status == 'no_text'
    monkeypatch.setattr('app.services.ocr.subprocess.run', lambda *args,**kwargs: subprocess.CompletedProcess([],0,b'Quantum\x00 text',b''))
    assert extract_image_text(image).text == 'Quantum text'
    monkeypatch.setattr('app.services.ocr.subprocess.run', lambda *args,**kwargs: subprocess.CompletedProcess([],1,b'',b'language missing'))
    assert extract_image_text(image).status == 'failed'


@pytest.mark.skipif(shutil.which('tesseract') is None, reason='Tesseract binary required for real OCR test')
def test_real_tesseract_image_upload_and_search(client):
    r = client.post('/api/v1/documents/upload-image', files={'file':('ocr-test.png',image_bytes(text='QUANTUM RETRIEVAL\nIMAGE OCR TEST 4271'))})
    assert r.status_code == 200, r.text
    result = r.json()
    assert result['ocr']['status'] == 'extracted'
    assert 'QUANTUM RETRIEVAL' in result['ocr']['text'].upper()
    assert '4271' in result['ocr']['text']
    assert result['passages_created'] >= 1
    found = client.get('/api/v1/search/keyword?q=quantum').json()['results']
    assert found[0]['document_id'] == result['document_id']
    assert client.get(f"/api/v1/documents/{result['document_id']}/ocr").json()['results'][0]['text'] == result['ocr']['text']
