"""Human review (active-learning queue and labels) plus workspace analytics."""
import json
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import LABELS, RESULTS_DIR
from app.dependencies import get_db
from app.models import Document, Figure, Passage, ReviewEvent, ReviewLabel, SourceAsset
from app.services.learning import MODELS, fit_classifier, rank_candidates, training_data
from app.services.search import LATENCIES, item_payload

router = APIRouter(prefix="/api/v1", tags=["Review"])
ItemType = Literal["passage", "figure"]


@router.get("/review/queue")
def review_queue(item_type: ItemType = "passage", strategy: Literal["entropy", "margin", "random"] = "entropy",
                 limit: int = Query(10, ge=1, le=100), seed: int = 42,
                 exclude: list[int] = Query(default=[], max_length=500), db: Session = Depends(get_db)):
    model = MODELS[item_type]
    rows = training_data(db, item_type)
    classifier = fit_classifier(rows)
    labeled = select(ReviewLabel.item_id).where(ReviewLabel.item_type == item_type)
    # Acquisition scores use only frozen embeddings, never unknown labels.
    items = db.scalars(select(model).where(model.embedding.is_not(None), ~model.id.in_(labeled),
                                           ~model.id.in_(exclude)).order_by(model.id)).all()
    ranked = rank_candidates(items, classifier, strategy, seed)[:limit]
    documents = {d.id: d for d in db.scalars(select(Document).where(Document.id.in_({i.document_id for i, *_ in ranked})))}
    return {"strategy": strategy if classifier is not None else "random_cold_start",
            "classifier_ready": classifier is not None, "labeled_count": len(rows), "remaining": len(items),
            "classes": classifier.classes_.tolist() if classifier is not None else [],
            "results": [{**item_payload(item, item_type, documents[item.document_id]), "uncertainty": score,
                         "prediction": prediction, "probabilities": probabilities}
                        for item, score, prediction, probabilities in ranked]}


class ReviewBody(BaseModel):
    item_type: ItemType
    item_id: int = Field(gt=0)
    label: Literal[LABELS]
    note: str = Field("", max_length=2000)


@router.post("/review/labels")
def label_item(body: ReviewBody, db: Session = Depends(get_db)):
    """Set the item's current label and append an audit event."""
    if db.get(MODELS[body.item_type], body.item_id) is None:
        raise HTTPException(404, "Item not found")
    current = db.scalar(select(ReviewLabel).where(ReviewLabel.item_type == body.item_type,
                                                  ReviewLabel.item_id == body.item_id))
    if current is None:
        db.add(ReviewLabel(**body.model_dump()))
    else:
        current.label, current.note = body.label, body.note
    db.add(ReviewEvent(**body.model_dump()))
    db.commit()
    classifier = fit_classifier(rows := training_data(db, body.item_type))
    return {"saved": True, "classifier_ready": classifier is not None, "training_examples": len(rows),
            "classes": classifier.classes_.tolist() if classifier is not None else []}


@router.get("/review/labels")
def list_labels(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    labels = db.scalars(select(ReviewLabel).order_by(ReviewLabel.updated_at.desc()).limit(limit))
    return [{"item_type": row.item_type, "item_id": row.item_id, "label": row.label, "note": row.note,
             "updated_at": row.updated_at} for row in labels]


@router.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    tables = {"documents": Document, "passages": Passage, "figures": Figure, "labels": ReviewLabel,
              "review_actions": ReviewEvent, "sources": SourceAsset}
    successful = [duration for _, duration, status in LATENCIES if status < 400]
    return {"counts": {name: db.scalar(select(func.count()).select_from(model)) for name, model in tables.items()},
            "label_counts": dict(db.execute(select(ReviewLabel.label, func.count()).group_by(ReviewLabel.label)).all()),
            "allowed_labels": LABELS,
            "search_latency": {"scope": "last 1000 search/chat requests in this server process",
                               "sample_count": len(successful),
                               "failed_requests": sum(status >= 400 for _, _, status in LATENCIES),
                               "median_ms": float(np.median(successful)) if successful else None,
                               "p95_ms": float(np.percentile(successful, 95)) if successful else None}}


@router.get("/experiments")
def experiment_results():
    reports = {}
    for name in ("retrieval_summary", "active_learning", "index_benchmark"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            try:
                reports[name] = json.loads(path.read_text())
            except (OSError, ValueError):
                reports[name] = {"error": "Report could not be read; rerun the experiment"}
    return {"reports": reports,
            "note": "Missing reports mean experiments have not been run. Provisional judgments require human verification."}
