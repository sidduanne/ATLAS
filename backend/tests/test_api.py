"""API and ingestion tests against an isolated PostgreSQL schema with deterministic fake encoders."""
import io
import shutil
import subprocess

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import func, select

from app.core import data_path
from app.models import Document, Figure, ImageText, Passage, ReviewEvent, ReviewLabel, SourceAsset
from app.services.ingestion import OCRResult, extract_image_text, split_text, store_figures
from app.services.learning import uncertainty
from conftest import vector


# --- Documents, search and review ---------------------------------------------------------------

def pdf_bytes(with_image=False):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Active learning prioritizes uncertain examples for human review. " * 4)
    if with_image:
        data = io.BytesIO()
        Image.new("RGB", (100, 100), "red").save(data, format="PNG")
        page.insert_image(pymupdf.Rect(50, 100, 250, 300), stream=data.getvalue())
    payload = doc.tobytes()
    doc.close()
    return payload


def test_pdf_ingestion_persists_source_passages_and_figures(client, db):
    response = client.post('/api/v1/documents/upload', files={'file': ('paper.pdf', pdf_bytes(True), 'application/pdf')})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['passages_created'] >= 1
    assert payload['figures_created'] >= 1
    did = payload['document_id']
    assert client.get(f'/api/v1/documents/{did}/source').content.startswith(b'%PDF')
    assert client.get('/api/v1/documents').json()[0]['has_source'] is True
    figures = client.get(f'/api/v1/documents/{did}/figures').json()
    assert client.get(figures[0]['preview_url']).headers['content-type'] == 'image/png'
    assert client.get('/api/v1/search/images', params={'q': 'red diagram'}).json()['results'][0]['figure_id'] == figures[0]['item_id']
    for endpoint in ['/api/v1/search', '/api/v1/search/keyword', '/api/v1/search/hybrid']:
        result = client.get(endpoint, params={'q': 'active learning'})
        assert result.status_code == 200, result.text
        assert result.json()['results'][0]['document_id'] == did
        assert 'server-timing' in result.headers


def test_transaction_failure_removes_database_and_files(client, db, monkeypatch):
    import app.services.encoders
    def broken(_):
        raise RuntimeError('test embedding failure')
    monkeypatch.setattr(app.services.encoders, 'embed_texts', broken)
    r = client.post('/api/v1/documents/upload', files={'file': ('paper.pdf', pdf_bytes(), 'application/pdf')})
    assert r.status_code == 500
    assert db.scalar(select(func.count()).select_from(Document)) == 0
    assert not list(data_path('sources').rglob('*.pdf'))


def test_attach_legacy_pdf_preserves_passages_and_rejects_overwrite(client, db):
    doc = Document(filename='old.pdf', title='old', status='processed')
    db.add(doc); db.flush()
    passage = Passage(document_id=doc.id, content='Original passage', page_number=1, embedding=vector('old'))
    db.add(passage); db.commit()
    pid = passage.id
    assert client.get(f'/api/v1/documents/{doc.id}/source').status_code == 404
    assert client.post(f'/api/v1/documents/{doc.id}/source', files={'file': ('wrong.pdf', pdf_bytes())}).status_code == 422
    assert client.post(f'/api/v1/documents/{doc.id}/source', files={'file': ('old.pdf', b'bad PDF')}).status_code == 422
    r = client.post(f'/api/v1/documents/{doc.id}/source', files={'file': ('old.pdf', pdf_bytes(True))})
    assert r.status_code == 200, r.text
    assert db.get(Passage, pid).content == 'Original passage'
    assert client.post(f'/api/v1/documents/{doc.id}/source', files={'file': ('old.pdf', pdf_bytes())}).status_code == 422


def test_text_image_upload_and_provenance(client, db):
    r = client.post('/api/v1/documents/upload-text', json={'title': 'Notes', 'content': 'Technical active learning notes.', 'source_url': 'https://example.org/paper', 'license': 'CC BY 4.0'})
    assert r.status_code == 200, r.text
    did = r.json()['document_id']
    assert client.get(f'/api/v1/documents/{did}/source').text == 'Technical active learning notes.'
    assert client.patch(f'/api/v1/documents/{did}/provenance', json={'license': 'CC0'}).status_code == 200
    assert db.get(SourceAsset, did).license == 'CC0'
    image = io.BytesIO(); Image.new('RGB', (64, 64), 'blue').save(image, format='PNG')
    r = client.post('/api/v1/documents/upload-image', files={'file': ('diagram.png', image.getvalue())})
    assert r.status_code == 200, r.text
    assert r.json()['figures_created'] == 1
    assert client.post('/api/v1/documents/upload-image', files={'file': ('bad.png', b'bad image')}).status_code == 422


def test_review_cold_start_training_corrections_and_modality_isolation(client, db):
    for i in range(8):
        doc = Document(filename=f'{i}.txt', title=str(i), status='processed')
        db.add(doc); db.flush()
        db.add(Passage(document_id=doc.id, content=f'Item {i}', page_number=1, embedding=vector(str(i))))
    db.commit()
    queue = client.get('/api/v1/review/queue').json()
    assert not queue['classifier_ready']
    assert queue['strategy'] == 'random_cold_start'
    ids = [i['item_id'] for i in queue['results']]
    for i, pid in enumerate(ids[:4]):
        r = client.post('/api/v1/review/labels', json={'item_type': 'passage', 'item_id': pid, 'label': 'relevant' if i % 2 else 'irrelevant'})
        assert r.status_code == 200, r.text
    assert r.json()['classifier_ready']
    for strategy in ['entropy', 'margin', 'random']:
        queue = client.get('/api/v1/review/queue', params={'strategy': strategy}).json()
        assert queue['classifier_ready']
        assert queue['remaining'] == 4
        assert not set(i['item_id'] for i in queue['results']) & set(ids[:4])
        if strategy != 'random':
            scores = [i['uncertainty'] for i in queue['results']]
            assert scores == sorted(scores, reverse=True)
    r = client.post('/api/v1/review/labels', json={'item_type': 'passage', 'item_id': ids[0], 'label': 'duplicate', 'note': 'Corrected'})
    assert r.status_code == 200
    assert db.scalar(select(func.count()).select_from(ReviewLabel)) == 4
    assert db.scalar(select(func.count()).select_from(ReviewEvent)) == 5
    assert client.get('/api/v1/review/queue?item_type=figure').json()['labeled_count'] == 0
    assert client.get('/api/v1/analytics').json()['counts']['review_actions'] == 5


def test_validation_and_bulk_partial_success(client):
    assert client.post('/api/v1/documents/upload', files={'file': ('bad.pdf', b'not pdf')}).status_code == 422
    r = client.post('/api/v1/documents/bulk-upload', files=[('files', ('bad.txt', b'bad')), ('files', ('good.pdf', pdf_bytes()))])
    assert r.status_code == 200, r.text
    assert r.json()['succeeded'] == 1 and r.json()['failed'] == 1
    assert client.post('/api/v1/review/labels', json={'item_type': 'passage', 'item_id': 9999, 'label': 'relevant'}).status_code == 404
    assert client.post('/api/v1/review/labels', json={'item_type': 'passage', 'item_id': 1, 'label': 'invalid'}).status_code == 422
    assert client.get('/api/v1/search?q=%20%20%20').status_code == 422
    assert client.get('/api/v1/review/queue?item_type=bad').status_code == 422
    assert client.post('/api/v1/documents/upload-text', json={'title': 'x', 'content': '   '}).status_code == 422


def test_conversation_preserves_refinement_and_source_citations(client):
    client.post('/api/v1/documents/upload-text', json={'title': 'Paper', 'content': 'Active learning uses uncertainty sampling.'})
    r = client.post('/api/v1/chat', json={'message': 'uncertainty sampling', 'history': ['active learning'], 'method': 'hybrid'})
    assert r.status_code == 200, r.text
    assert r.json()['query'] == 'active learning uncertainty sampling'
    assert r.json()['response_type'] == 'retrieval_refinement'
    assert r.json()['results'][0]['document_id']
    assert client.post('/api/v1/chat', json={'message': '   '}).status_code == 422


def test_uncertainty_and_split_boundaries():
    p = np.array([[0.5, 0.5], [0.99, 0.01]])
    assert uncertainty(p)[0] > uncertainty(p)[1]
    assert uncertainty(p, 'margin')[0] > uncertainty(p, 'margin')[1]
    assert split_text('one two three four five', chunk_size=10, overlap=3)
    for overlap in [-1, 10]:
        with pytest.raises(ValueError):
            split_text('test', chunk_size=10, overlap=overlap)


def test_real_frozen_models_and_multimodal_ingestion(db):
    """A separate real-model test in an isolated schema, with no mocked encoders."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.core import get_db
    from app.services.encoders import embed_image_query, embed_images, embed_query
    image = Image.new('RGB', (80, 80), 'red')
    a = np.array(embed_query('active learning'))
    b = np.array(embed_image_query('red square'))
    c = np.array(embed_images([image])[0])
    assert a.shape == (384,) and b.shape == c.shape == (512,)
    assert np.isclose(np.linalg.norm(a), 1, atol=1e-4)
    assert np.isclose(np.linalg.norm(b), 1, atol=1e-4)
    def override():
        yield db
    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            r = client.post('/api/v1/documents/upload', files={'file': ('real-model-test.pdf', pdf_bytes(True), 'application/pdf')})
            assert r.status_code == 200, r.text
            assert r.json()['figures_created'] == 1
            r = client.get('/api/v1/search/images?q=red%20square')
            assert r.status_code == 200 and len(r.json()['results']) == 1
            r = client.get('/api/v1/search?q=active%20learning')
            assert r.status_code == 200 and len(r.json()['results']) >= 1
    finally:
        app.dependency_overrides.clear()


# --- Image OCR ---------------------------------------------------------------------------------

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
    monkeypatch.setattr('app.services.ingestion.extract_image_text', recognize)
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
    monkeypatch.setattr('app.services.ingestion.extract_image_text', lambda image: OCRResult('', 'unavailable', 'Install Tesseract'))
    result = client.post('/api/v1/documents/upload-image', files={'file':('scan.png',image_bytes())}).json()
    assert result['ocr']['status'] == 'unavailable' and result['passages_created'] == 0
    fid = result['ocr']['figure_id']
    assert client.get('/api/v1/search/images?q=diagram').json()['results']
    monkeypatch.setattr('app.services.ingestion.extract_image_text', lambda image: OCRResult('Recovered text', 'extracted'))
    assert client.post(f'/api/v1/figures/{fid}/ocr').json()['status'] == 'unavailable'
    assert client.post(f'/api/v1/figures/{fid}/ocr?retry=true').json()['text'] == 'Recovered text'
    assert db.scalar(select(func.count()).select_from(Passage)) == 1


def test_ocr_embedding_failure_rolls_back_image_and_files(client, db, monkeypatch):
    monkeypatch.setattr('app.services.ingestion.extract_image_text', lambda image: OCRResult('Recognized text', 'extracted'))
    def fail(_):
        raise RuntimeError('Embedding failed')
    monkeypatch.setattr('app.services.encoders.embed_texts', fail)
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
    monkeypatch.setattr('app.services.ingestion.subprocess.run', missing)
    assert extract_image_text(image).status == 'unavailable'
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('tesseract',60)
    monkeypatch.setattr('app.services.ingestion.subprocess.run', timeout)
    assert extract_image_text(image).status == 'timeout'
    monkeypatch.setattr('app.services.ingestion.subprocess.run', lambda *args,**kwargs: subprocess.CompletedProcess([],0,b'\x00  \n',b''))
    assert extract_image_text(image).status == 'no_text'
    monkeypatch.setattr('app.services.ingestion.subprocess.run', lambda *args,**kwargs: subprocess.CompletedProcess([],0,b'Quantum\x00 text',b''))
    assert extract_image_text(image).text == 'Quantum text'
    monkeypatch.setattr('app.services.ingestion.subprocess.run', lambda *args,**kwargs: subprocess.CompletedProcess([],1,b'',b'language missing'))
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


# --- Research workflow -------------------------------------------------------------------------

def add_passage(db):
    doc = Document(filename='research.txt', title='Research', status='processed')
    db.add(doc); db.flush()
    item = Passage(document_id=doc.id, content='Active learning selects uncertain examples for labeling.', page_number=1, embedding=vector('research'))
    db.add(item); db.commit()
    return item


def test_human_judgment_verification_and_invalidation(client, db):
    item = add_passage(db)
    q = client.post('/api/v1/research/queries', json={'query':'active learning', 'modality':'text'}).json()
    qid = q['id']
    assert client.post(f'/api/v1/research/queries/{qid}/verify').status_code == 422
    pool = client.post(f'/api/v1/research/queries/{qid}/pool').json()
    assert len(pool['candidates']) == 1 and pool['candidates'][0]['relevant'] is None
    assert client.post(f'/api/v1/research/queries/{qid}/judgments', json={'item_id':item.id,'relevant':True}).status_code == 200
    assert client.post(f'/api/v1/research/queries/{qid}/verify').json()['verified']
    exported = client.get('/api/v1/research/queries/export').json()
    assert exported['queries'][0]['relevant_passage_ids'] == [item.id]
    assert client.get('/api/v1/research/readiness').json()['verified_text_queries'] == 1
    client.post(f'/api/v1/research/queries/{qid}/judgments', json={'item_id':item.id,'relevant':False})
    assert client.get('/api/v1/research/queries/export').json()['queries'] == []
    assert client.post(f'/api/v1/research/queries/{qid}/verify').status_code == 422


def test_readiness_and_runs_block_without_human_ground_truth(client):
    state = client.get('/api/v1/research/readiness').json()
    assert not state['active_learning']['passage']['ready']
    assert state['active_learning']['passage']['labels'] == 0
    assert client.post('/api/v1/research/runs',json={'experiment':'retrieval'}).status_code == 422
    assert client.post('/api/v1/research/runs',json={'experiment':'active_learning'}).status_code == 422
    assert client.get('/api/v1/research/labels/export').json() == {'labels':[]}
    assert client.post('/api/v1/research/queries',json={'query':'   '}).status_code == 422


def test_experiment_run_uses_fixed_command_and_rejects_overlap(client, monkeypatch):
    import app.api.research as research
    submitted = []
    monkeypatch.setattr(research, 'jobs', {})
    monkeypatch.setattr(research.executor, 'submit', lambda *args: submitted.append(args))
    result = client.post('/api/v1/research/runs',json={'experiment':'index'})
    assert result.status_code == 200
    job = result.json()
    assert submitted[0][1] == job['id']
    assert submitted[0][2][-2:] == ['--repeats','10']
    assert client.get(f"/api/v1/research/runs/{job['id']}").json()['status'] == 'queued'
    assert client.post('/api/v1/research/runs',json={'experiment':'index'}).status_code == 409
    assert client.post('/api/v1/research/runs',json={'experiment':'../../bad'}).status_code == 422


def test_skip_does_not_create_a_label(client, db):
    item = add_passage(db)
    result = client.get('/api/v1/review/queue', params={'exclude':item.id})
    assert result.status_code == 200
    assert result.json()['results'] == []
    assert client.get('/api/v1/research/labels/export').json()['labels'] == []


def test_figure_caption_nul_is_sanitized(client, db):
    item = add_passage(db)
    created = []
    store_figures([(1, Image.new('RGB',(64,64),'blue'), 'broken\x00caption')], item.document_id, db, created)
    db.commit()
    assert db.scalar(select(Figure)).caption == 'brokencaption'
