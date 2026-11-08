from sqlalchemy import select
from app.models import Document, Passage
from app.models.review import EvaluationQuery
from conftest import vector


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
    from app.services.assets import store_figures
    from app.models import Figure
    from PIL import Image
    item = add_passage(db)
    created = []
    store_figures([(1, Image.new('RGB',(64,64),'blue'), 'broken\x00caption')], item.document_id, db, created)
    db.commit()
    assert db.scalar(select(Figure)).caption == 'brokencaption'
