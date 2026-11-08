"""Human relevance judging, experiment readiness, and bounded experiment execution."""
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import ROOT, RESULTS_DIR
from app.dependencies import get_db
from app.models import Document, Figure, Passage, ReviewLabel, SourceAsset
from app.models.review import EvaluationCandidate, EvaluationQuery
from app.services.evaluation import grouped_split, learning_dataset

router = APIRouter(prefix="/api/v1/research", tags=["Research evaluation"])
executor = ThreadPoolExecutor(max_workers=1)
lock = Lock()
jobs = {}


def query_payload(db, query):
    candidates = db.scalars(select(EvaluationCandidate).where(EvaluationCandidate.query_id == query.id).order_by(EvaluationCandidate.id)).all()
    model = Figure if query.modality == "image" else Passage
    output = []
    for candidate in candidates:
        item = db.get(model, candidate.item_id)
        if item is None:
            continue
        doc = db.get(Document, item.document_id)
        output.append({"item_id": item.id, "relevant": candidate.relevant, "document_id": doc.id,
                       "filename": doc.filename, "page_number": item.page_number,
                       "content": item.caption if query.modality == "image" else item.content,
                       "preview_url": f"/api/v1/figures/{item.id}/image" if query.modality == "image" else None})
    return {"id": query.id, "query": query.query, "modality": query.modality, "verified": query.verified, "candidates": output}


class NewQuery(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    modality: Literal["text", "image"] = "text"


@router.post("/queries")
def create_query(body: NewQuery, db: Session = Depends(get_db)):
    if len(body.query.strip()) < 3:
        raise HTTPException(422, "Enter a meaningful query")
    query = EvaluationQuery(query=body.query.strip(), modality=body.modality, verified=False)
    db.add(query); db.commit()
    return query_payload(db, query)


@router.get("/queries")
def list_queries(db: Session = Depends(get_db)):
    return [query_payload(db, q) for q in db.scalars(select(EvaluationQuery).order_by(EvaluationQuery.id))]


@router.post("/queries/{query_id}/pool")
def pool_candidates(query_id: int, db: Session = Depends(get_db)):
    query = db.get(EvaluationQuery, query_id)
    if query is None:
        raise HTTPException(404, "Query not found")
    if query.modality == "image":
        from app.api.workspace import image_search
        pools = [image_search(q=query.query, limit=5, db=db)["results"]]
        key = "figure_id"
    else:
        from app.main import semantic_search
        from app.api.keyword_search import keyword_search
        from app.api.hybrid_search import hybrid_search
        pools = [method(q=query.query, limit=5, db=db)["results"] for method in (semantic_search, keyword_search, hybrid_search)]
        key = "passage_id"
    ids = sorted({item[key] for pool in pools for item in pool})
    existing = set(db.scalars(select(EvaluationCandidate.item_id).where(EvaluationCandidate.query_id == query.id)))
    for item_id in ids:
        if item_id not in existing:
            db.add(EvaluationCandidate(query_id=query.id, item_id=item_id, relevant=None))
    if set(ids) - existing:
        query.verified = False
    db.commit()
    return query_payload(db, query)


class Judgment(BaseModel):
    item_id: int = Field(gt=0)
    relevant: bool


@router.post("/queries/{query_id}/judgments")
def judge(query_id: int, body: Judgment, db: Session = Depends(get_db)):
    query = db.get(EvaluationQuery, query_id)
    candidate = db.scalar(select(EvaluationCandidate).where(EvaluationCandidate.query_id == query_id, EvaluationCandidate.item_id == body.item_id))
    if query is None or candidate is None:
        raise HTTPException(404, "Query candidate not found")
    candidate.relevant = body.relevant
    query.verified = False
    db.commit()
    return query_payload(db, query)


@router.post("/queries/{query_id}/verify")
def verify(query_id: int, db: Session = Depends(get_db)):
    query = db.get(EvaluationQuery, query_id)
    if query is None:
        raise HTTPException(404, "Query not found")
    payload = query_payload(db, query)
    candidates = payload["candidates"]
    if not candidates or any(c["relevant"] is None for c in candidates) or not any(c["relevant"] is True for c in candidates):
        raise HTTPException(422, "Judge every pooled candidate and identify at least one relevant result before verifying")
    query.verified = True
    db.commit()
    return query_payload(db, query)


@router.get("/queries/export")
def export_queries(db: Session = Depends(get_db)):
    output = []
    for query in db.scalars(select(EvaluationQuery).where(EvaluationQuery.verified.is_(True)).order_by(EvaluationQuery.id)):
        payload = query_payload(db, query)
        key = "relevant_figure_ids" if query.modality == "image" else "relevant_passage_ids"
        output.append({"id": f"Q{query.id}", "query": query.query, "modality": query.modality,
                       key: [c["item_id"] for c in payload["candidates"] if c["relevant"] is True]})
    return {"judgments_verified": True, "queries": output,
            "notes": "Human-verified pooled judgments. Recall is relative to the judged pool; relevant items outside the pool may be missing."}


@router.get("/labels/export")
def export_labels(db: Session = Depends(get_db)):
    return {"labels": [{"item_type": label.item_type, "item_id": label.item_id, "label": label.label,
                        "note": label.note, "updated_at": label.updated_at.isoformat()} for label in db.scalars(select(ReviewLabel).order_by(ReviewLabel.id))]}


@router.get("/readiness")
def readiness(db: Session = Depends(get_db)):
    def count(model):
        return db.scalar(select(func.count()).select_from(model))
    modalities = {}
    for kind in ("passage", "figure"):
        x, y, ids, groups = learning_dataset(db, kind)
        problem = None
        train_size = 0
        if len(y) < 12 or len(set(y)) < 2:
            problem = "Label at least 12 items across at least 2 classes from multiple independent documents. Aim for 50–100+ for useful curves."
        else:
            try:
                train, test, _ = grouped_split(y, groups)
                train_size = len(train)
            except ValueError as exc:
                problem = str(exc)
        modalities[kind] = {"labels": len(y), "classes": dict(Counter(y)), "documents_or_duplicate_groups": len(set(groups)),
                            "ready": problem is None, "blocking_reason": problem, "max_budget": train_size}
    return {"documents": count(Document), "sources": count(SourceAsset), "figures": count(Figure),
            "sources_missing": count(Document)-count(SourceAsset),
            "licenses_missing": db.scalar(select(func.count()).select_from(SourceAsset).where((SourceAsset.license.is_(None)) | (SourceAsset.license == ""))),
            "verified_text_queries": db.scalar(select(func.count()).select_from(EvaluationQuery).where(EvaluationQuery.verified.is_(True), EvaluationQuery.modality == "text")),
            "verified_image_queries": db.scalar(select(func.count()).select_from(EvaluationQuery).where(EvaluationQuery.verified.is_(True), EvaluationQuery.modality == "image")),
            "active_learning": modalities}


class RunExperiment(BaseModel):
    experiment: Literal["retrieval", "active_learning", "index"]
    item_type: Literal["passage", "figure"] = "passage"


def execute_job(job_id, command):
    with lock:
        jobs[job_id]["status"] = "running"
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "backend"))
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        with lock:
            jobs[job_id].update(status="completed" if result.returncode == 0 else "failed",
                                output=(result.stdout + "\n" + result.stderr)[-12000:], exit_code=result.returncode)
    except Exception as exc:
        with lock:
            jobs[job_id].update(status="failed", output=str(exc))


@router.post("/runs")
def run_experiment(body: RunExperiment, db: Session = Depends(get_db)):
    dataset = None
    if body.experiment == "retrieval":
        dataset = export_queries(db)
        if not dataset["queries"]:
            raise HTTPException(422, "Create and verify relevance judgments first")
    if body.experiment == "active_learning":
        state = readiness(db)["active_learning"][body.item_type]
        if not state["ready"]:
            raise HTTPException(422, state["blocking_reason"])
    job_id = uuid4().hex
    scripts = {"retrieval": "evaluate_retrieval.py", "active_learning": "evaluate_active_learning.py", "index": "benchmark_index.py"}
    command = [sys.executable, str(ROOT / "experiments" / scripts[body.experiment])]
    if body.experiment == "retrieval":
        # Each run gets immutable human-judgment input; never overwrite the starter dataset.
        command += ["--dataset", str(RESULTS_DIR / f"judgments_{job_id}.json")]
    elif body.experiment == "active_learning":
        command += ["--item-type", body.item_type]
    else:
        command += ["--repeats", "10"]
    with lock:
        if any(job["status"] in {"queued", "running"} for job in jobs.values()):
            raise HTTPException(409, "An experiment is already running")
        if dataset is not None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            (RESULTS_DIR / f"judgments_{job_id}.json").write_text(json.dumps(dataset, indent=2))
        if len(jobs) >= 30:
            jobs.pop(next(iter(jobs)))
        jobs[job_id] = {"id": job_id, "experiment": body.experiment, "status": "queued", "output": ""}
    executor.submit(execute_job, job_id, command)
    return dict(jobs[job_id])


@router.get("/runs/{job_id}")
def job_status(job_id: str):
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, "Run unavailable; run tracking resets on backend restart")
        return dict(jobs[job_id])
