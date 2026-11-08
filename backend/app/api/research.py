"""Human relevance judging, experiment readiness, and bounded background experiment runs."""
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core import RESULTS_DIR, ROOT, get_db
from app.models import Document, EvaluationCandidate, EvaluationQuery, Figure, Passage, ReviewLabel, SourceAsset
from app.services import search
from app.services.learning import grouped_split, learning_dataset

router = APIRouter(prefix="/api/v1/research", tags=["Research evaluation"])

# Experiment jobs: one at a time, tracked in this process only (reset on restart).
executor = ThreadPoolExecutor(max_workers=1)
lock = Lock()
jobs: dict[str, dict] = {}
SCRIPTS = {"retrieval": "evaluate_retrieval.py", "active_learning": "evaluate_active_learning.py",
           "index": "benchmark_index.py"}


def get_query(db, query_id) -> EvaluationQuery:
    query = db.get(EvaluationQuery, query_id)
    if query is None:
        raise HTTPException(404, "Query not found")
    return query


def query_payload(db, query) -> dict:
    kind, model = ("figure", Figure) if query.modality == "image" else ("passage", Passage)
    candidates = []
    for candidate in db.scalars(select(EvaluationCandidate).where(EvaluationCandidate.query_id == query.id)
                                .order_by(EvaluationCandidate.id)):
        item = db.get(model, candidate.item_id)
        if item is not None:
            candidates.append({**search.item_payload(item, kind, db.get(Document, item.document_id)),
                               "relevant": candidate.relevant})
    return {"id": query.id, "query": query.query, "modality": query.modality, "verified": query.verified,
            "candidates": candidates}


class NewQuery(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    modality: Literal["text", "image"] = "text"


@router.post("/queries")
def create_query(body: NewQuery, db: Session = Depends(get_db)):
    query = EvaluationQuery(query=body.query, modality=body.modality, verified=False)
    db.add(query)
    db.commit()
    return query_payload(db, query)


@router.get("/queries")
def list_queries(db: Session = Depends(get_db)):
    return [query_payload(db, q) for q in db.scalars(select(EvaluationQuery).order_by(EvaluationQuery.id))]


@router.post("/queries/{query_id}/pool")
def pool_candidates(query_id: int, db: Session = Depends(get_db)):
    """Add the top 5 results of every applicable method as unjudged candidates."""
    query = get_query(db, query_id)
    methods, key = ((search.images,), "figure_id") if query.modality == "image" else \
        ((search.semantic, search.keyword, search.hybrid), "passage_id")
    ids = {result[key] for method in methods for result in method(db, query.query, 5)}
    existing = set(db.scalars(select(EvaluationCandidate.item_id).where(EvaluationCandidate.query_id == query.id)))
    for item_id in sorted(ids - existing):
        db.add(EvaluationCandidate(query_id=query.id, item_id=item_id, relevant=None))
    if ids - existing:
        query.verified = False
    db.commit()
    return query_payload(db, query)


class Judgment(BaseModel):
    item_id: int = Field(gt=0)
    relevant: bool


@router.post("/queries/{query_id}/judgments")
def judge(query_id: int, body: Judgment, db: Session = Depends(get_db)):
    query = db.get(EvaluationQuery, query_id)
    candidate = db.scalar(select(EvaluationCandidate).where(EvaluationCandidate.query_id == query_id,
                                                            EvaluationCandidate.item_id == body.item_id))
    if query is None or candidate is None:
        raise HTTPException(404, "Query candidate not found")
    candidate.relevant = body.relevant
    query.verified = False  # Any change needs a fresh human verification.
    db.commit()
    return query_payload(db, query)


@router.post("/queries/{query_id}/verify")
def verify(query_id: int, db: Session = Depends(get_db)):
    query = get_query(db, query_id)
    judged = [c["relevant"] for c in query_payload(db, query)["candidates"]]
    if not judged or None in judged or True not in judged:
        raise HTTPException(422, "Judge every pooled candidate and identify at least one relevant result before verifying")
    query.verified = True
    db.commit()
    return query_payload(db, query)


@router.get("/queries/export")
def export_queries(db: Session = Depends(get_db)):
    output = []
    for query in db.scalars(select(EvaluationQuery).where(EvaluationQuery.verified.is_(True)).order_by(EvaluationQuery.id)):
        key = "relevant_figure_ids" if query.modality == "image" else "relevant_passage_ids"
        relevant = [c["item_id"] for c in query_payload(db, query)["candidates"] if c["relevant"] is True]
        output.append({"id": f"Q{query.id}", "query": query.query, "modality": query.modality, key: relevant})
    return {"judgments_verified": True, "queries": output,
            "notes": "Human-verified pooled judgments. Recall is relative to the judged pool; "
                     "relevant items outside the pool may be missing."}


@router.get("/labels/export")
def export_labels(db: Session = Depends(get_db)):
    return {"labels": [{"item_type": label.item_type, "item_id": label.item_id, "label": label.label,
                        "note": label.note, "updated_at": label.updated_at.isoformat()}
                       for label in db.scalars(select(ReviewLabel).order_by(ReviewLabel.id))]}


def active_learning_state(db, kind) -> dict:
    _, y, _, groups = learning_dataset(db, kind)
    problem, train_size = None, 0
    if len(y) < 12 or len(set(y)) < 2:
        problem = ("Label at least 12 items across at least 2 classes from multiple independent documents. "
                   "Aim for 50–100+ for useful curves.")
    else:
        try:
            train_size = len(grouped_split(y, groups)[0])
        except ValueError as exc:
            problem = str(exc)
    return {"labels": len(y), "classes": dict(Counter(y)), "documents_or_duplicate_groups": len(set(groups)),
            "ready": problem is None, "blocking_reason": problem, "max_budget": train_size}


@router.get("/readiness")
def readiness(db: Session = Depends(get_db)):
    def count(model, *where):
        return db.scalar(select(func.count()).select_from(model).where(*where))
    verified = EvaluationQuery.verified.is_(True)
    return {"documents": count(Document), "sources": count(SourceAsset), "figures": count(Figure),
            "sources_missing": count(Document) - count(SourceAsset),
            "licenses_missing": count(SourceAsset, SourceAsset.license.is_(None) | (SourceAsset.license == "")),
            "verified_text_queries": count(EvaluationQuery, verified, EvaluationQuery.modality == "text"),
            "verified_image_queries": count(EvaluationQuery, verified, EvaluationQuery.modality == "image"),
            "active_learning": {kind: active_learning_state(db, kind) for kind in ("passage", "figure")}}


class RunExperiment(BaseModel):
    experiment: Literal["retrieval", "active_learning", "index"]
    item_type: Literal["passage", "figure"] = "passage"


def execute_job(job_id, command):
    with lock:
        jobs[job_id]["status"] = "running"
    try:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "backend")),
                                capture_output=True, text=True, timeout=1800)
        update = {"status": "completed" if result.returncode == 0 else "failed", "exit_code": result.returncode,
                  "output": (result.stdout + "\n" + result.stderr)[-12000:]}
    except Exception as exc:
        update = {"status": "failed", "output": str(exc)}
    with lock:
        jobs[job_id].update(update)


@router.post("/runs")
def run_experiment(body: RunExperiment, db: Session = Depends(get_db)):
    """Start one allow-listed experiment script in the background."""
    job_id = uuid4().hex
    command = [sys.executable, str(ROOT / "experiments" / SCRIPTS[body.experiment])]
    dataset = None
    if body.experiment == "retrieval":
        dataset = export_queries(db)
        if not dataset["queries"]:
            raise HTTPException(422, "Create and verify relevance judgments first")
        # Each run gets an immutable snapshot of the human judgments.
        command += ["--dataset", str(RESULTS_DIR / f"judgments_{job_id}.json")]
    elif body.experiment == "active_learning":
        state = active_learning_state(db, body.item_type)
        if not state["ready"]:
            raise HTTPException(422, state["blocking_reason"])
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
