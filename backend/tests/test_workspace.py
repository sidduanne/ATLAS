import io

import numpy as np
import pymupdf
import pytest
from PIL import Image
from sqlalchemy import func, select

from app.config import data_path
from app.models import Document, Figure, Passage, ReviewEvent, ReviewLabel, SourceAsset
from app.services.ingestion import split_text
from app.services.learning import uncertainty
from conftest import vector


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
    import app.services.ingestion
    def broken(_):
        raise RuntimeError('test embedding failure')
    monkeypatch.setattr(app.services.ingestion, 'embed_texts', broken)
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
    from app.dependencies import get_db
    from app.services.clip import embed_image_query, embed_images
    from app.services.embeddings import embed_query
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
