import io
import json
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Literal

import numpy as np
import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import ALLOWED_LABELS, RESULTS_DIR, data_path
from app.dependencies import get_db
from app.models import Document, Figure, Passage, ReviewEvent, ReviewLabel, SourceAsset, ImageText
from app.services.assets import attach_pdf, save_source, store_figures
from app.services.image_text import index_image_text, ocr_payload
from app.services.clip import embed_image_query
from app.services.embeddings import embed_texts
from app.services.ingestion import split_text
from app.services.learning import MODELS, fit_classifier, rank_candidates, training_data

router = APIRouter(prefix="/api/v1", tags=["Workspace"])
LATENCIES = deque(maxlen=1000)
ItemType = Literal["passage", "figure"]
Label = Literal["relevant", "duplicate", "malformed", "out_of_domain", "irrelevant"]


def require_item(db, item_type, item_id):
    item = db.get(MODELS[item_type], item_id)
    if item is None:
        raise HTTPException(404, "Item not found")
    return item


def serialize_item(item, item_type, document):
    return {"item_id": item.id, "item_type": item_type, "document_id": item.document_id,
            "filename": document.filename, "page_number": item.page_number,
            "content": item.content if item_type == "passage" else item.caption,
            "preview_url": f"/api/v1/figures/{item.id}/image" if item_type == "figure" else None,
            "source_url": f"/api/v1/documents/{item.document_id}/source"}


@router.get("/search/images")
def image_search(q: str = Query(min_length=2, max_length=500), limit: int = Query(5, ge=1, le=50), db: Session = Depends(get_db)):
    if not q.strip():
        raise HTTPException(422, "Search query cannot be empty")
    distance = Figure.embedding.cosine_distance(embed_image_query(q))
    rows = db.execute(select(Figure, Document, distance.label("distance")).join(
        Document, Figure.document_id == Document.id).where(Figure.embedding.is_not(None))
        .order_by(distance).limit(limit)).all()
    return {"query": q, "method": "clip", "results": [
        {**serialize_item(item, "figure", doc), "figure_id": item.id, "similarity": 1-float(d)} for item, doc, d in rows]}


@router.get("/documents/{document_id}/figures")
def document_figures(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return [serialize_item(item, "figure", doc) for item in db.scalars(
        select(Figure).where(Figure.document_id == document_id).order_by(Figure.id))]


@router.get("/figures/{figure_id}/image")
def figure_image(figure_id: int, db: Session = Depends(get_db)):
    item = require_item(db, "figure", figure_id)
    path = data_path(item.storage_path)
    if not path.is_file():
        raise HTTPException(404, "Figure file unavailable")
    return FileResponse(path, media_type="image/png")


@router.get("/documents/{document_id}/source")
def document_source(document_id: int, db: Session = Depends(get_db)):
    source = db.get(SourceAsset, document_id)
    if not source or not data_path(source.storage_path).is_file():
        raise HTTPException(404, "Source unavailable. Attach the original file to this legacy record.")
    return FileResponse(data_path(source.storage_path), media_type=source.media_type,
                        headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})


@router.post("/documents/{document_id}/source")
def attach_source(document_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    try:
        if document is None:
            raise HTTPException(404, "Document not found")
        if Path(file.filename or "").name != document.filename or not document.filename.lower().endswith(".pdf"):
            raise HTTPException(422, "Select the original PDF with the same filename as this document")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pdf"
            with path.open("wb") as output:
                size = 0
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 500 * 1024 * 1024:
                        raise HTTPException(413, "PDF exceeds 500 MB")
                    output.write(chunk)
            try:
                with pymupdf.open(path) as pdf:
                    if not pdf.is_pdf or pdf.needs_pass:
                        raise ValueError("Attach a valid PDF that does not require a password")
            except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
                raise ValueError("The attached PDF is empty or malformed") from exc
            return attach_pdf(path, document, db)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc
    finally:
        file.file.close()


class Provenance(BaseModel):
    source_url: HttpUrl | None = None
    license: str | None = Field(None, max_length=200)


@router.patch("/documents/{document_id}/provenance")
def update_provenance(document_id: int, body: Provenance, db: Session = Depends(get_db)):
    source = db.get(SourceAsset, document_id)
    if source is None:
        raise HTTPException(404, "Attach a source file first")
    source.source_url = str(body.source_url) if body.source_url else None
    source.license = body.license
    db.commit()
    return {"document_id": document_id, "source_url": source.source_url, "license": source.license}


class TextDocument(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=2, max_length=1_000_000)
    source_url: HttpUrl | None = None
    license: str | None = Field(None, max_length=200)


@router.post("/documents/upload-text")
def upload_text(body: TextDocument, db: Session = Depends(get_db)):
    chunks = split_text(body.content)
    if not chunks:
        raise HTTPException(422, "Text cannot be blank")
    doc = Document(filename=Path(body.title).name[:200] + ".txt", title=body.title, status="processing")
    created = []
    try:
        db.add(doc)
        db.flush()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.txt"
            path.write_text(body.content, encoding="utf-8")
            source = save_source(path, doc.id, "text/plain", str(body.source_url) if body.source_url else None, body.license)
        created.append(data_path(source.storage_path))
        db.add(source)
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start+32]
            for chunk, vector in zip(batch, embed_texts(batch), strict=True):
                db.add(Passage(document_id=doc.id, content=chunk, page_number=1, embedding=vector))
        doc.status = "processed"
        db.commit()
        return {"document_id": doc.id, "passages_created": len(chunks), "status": doc.status}
    except Exception:
        db.rollback()
        for path in created:
            path.unlink(missing_ok=True)
        raise


@router.post("/documents/upload-image")
def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    created = []
    try:
        content = file.file.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(413, "Image exceeds 20 MB")
        image = Image.open(io.BytesIO(content))
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise HTTPException(422, "Use a PNG, JPEG or WebP image")
        if image.width * image.height > 20_000_000:
            raise HTTPException(413, "Images must be at most 20 megapixels")
        image.load()
        image = ImageOps.exif_transpose(image)
        original = image.copy()
        image = image.convert("RGB")
        image.thumbnail((1600, 1600))
        filename = Path(file.filename or "image.png").name[:255]
        doc = Document(filename=filename, title=Path(filename).stem, status="processed")
        db.add(doc)
        db.flush()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            original.save(path, format="PNG")
            source = save_source(path, doc.id, "image/png")
        created.append(data_path(source.storage_path))
        db.add(source)
        store_figures([(1, image, filename)], doc.id, db, created)
        db.flush()
        figure = db.scalar(select(Figure).where(Figure.document_id == doc.id))
        ocr = index_image_text(figure, original, db, embed_texts)
        db.flush()
        passages = db.scalar(select(func.count()).select_from(Passage).where(Passage.document_id == doc.id))
        db.commit()
        return {"document_id": doc.id, "figures_created": 1, "passages_created": passages,
                "ocr": ocr_payload(ocr), "status": "processed"}
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        db.rollback()
        for path in created:
            path.unlink(missing_ok=True)
        raise HTTPException(422, "Image cannot be decoded safely") from exc
    except Exception:
        db.rollback()
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()


@router.get("/review/queue")
def review_queue(item_type: ItemType = "passage", strategy: Literal["entropy", "margin", "random"] = "entropy",
                 limit: int = Query(10, ge=1, le=100), seed: int = 42, exclude: list[int] = Query(default=[], max_length=500), db: Session = Depends(get_db)):
    model = MODELS[item_type]
    rows = training_data(db, item_type)
    classifier = fit_classifier(rows)
    labeled = select(ReviewLabel.item_id).where(ReviewLabel.item_type == item_type)
    # Acquisition scores use only frozen embeddings, never unknown labels.
    items = db.scalars(select(model).where(model.embedding.is_not(None), ~model.id.in_(labeled), ~model.id.in_(exclude)).order_by(model.id)).all()
    ranked = rank_candidates(items, classifier, strategy, seed)[:limit]
    documents = {d.id: d for d in db.scalars(select(Document).where(Document.id.in_({i.document_id for i, *_ in ranked})))}
    return {"strategy": strategy if classifier is not None else "random_cold_start",
            "classifier_ready": classifier is not None, "labeled_count": len(rows), "remaining": len(items),
            "classes": classifier.classes_.tolist() if classifier is not None else [],
            "results": [{**serialize_item(item, item_type, documents[item.document_id]), "uncertainty": score,
                         "prediction": prediction, "probabilities": probabilities} for item, score, prediction, probabilities in ranked]}


class ReviewBody(BaseModel):
    item_type: ItemType
    item_id: int = Field(gt=0)
    label: Label
    note: str = Field("", max_length=2000)


@router.post("/review/labels")
def label_item(body: ReviewBody, db: Session = Depends(get_db)):
    require_item(db, body.item_type, body.item_id)
    current = db.scalar(select(ReviewLabel).where(ReviewLabel.item_type == body.item_type, ReviewLabel.item_id == body.item_id))
    if current is None:
        current = ReviewLabel(**body.model_dump())
        db.add(current)
    else:
        current.label = body.label
        current.note = body.note
    db.add(ReviewEvent(**body.model_dump()))
    db.commit()
    # Fit now, so corrections immediately affect predictions and the next queue.
    rows = training_data(db, body.item_type)
    classifier = fit_classifier(rows)
    return {"saved": True, "classifier_ready": classifier is not None, "training_examples": len(rows),
            "classes": classifier.classes_.tolist() if classifier is not None else []}


@router.get("/review/labels")
def list_labels(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    return [{"item_type": row.item_type, "item_id": row.item_id, "label": row.label,
             "note": row.note, "updated_at": row.updated_at} for row in db.scalars(
                 select(ReviewLabel).order_by(ReviewLabel.updated_at.desc()).limit(limit))]


@router.get("/analytics")
def analytics(db: Session = Depends(get_db)):
    counts = {name: db.scalar(select(func.count()).select_from(model)) for name, model in
              [("documents", Document), ("passages", Passage), ("figures", Figure), ("labels", ReviewLabel), ("review_actions", ReviewEvent), ("sources", SourceAsset)]}
    label_counts = dict(db.execute(select(ReviewLabel.label, func.count()).group_by(ReviewLabel.label)).all())
    successful = [duration for _, duration, status in LATENCIES if status < 400]
    return {"counts": counts, "label_counts": label_counts, "allowed_labels": ALLOWED_LABELS,
            "search_latency": {"scope": "last 1000 search/chat requests in this server process", "sample_count": len(successful),
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
    return {"reports": reports, "note": "Missing reports mean experiments have not been run. Provisional judgments require human verification."}


class ChatBody(BaseModel):
    message: str = Field(min_length=2, max_length=500)
    history: list[str] = Field(default_factory=list, max_length=10)
    method: Literal["semantic", "keyword", "hybrid", "images"] = "hybrid"


@router.post("/chat")
def chat(body: ChatBody, db: Session = Depends(get_db)):
    if not body.message.strip() or any(len(turn) > 500 for turn in body.history):
        raise HTTPException(422, "Each turn must contain at most 500 characters and the message cannot be blank")
    query = " ".join([*body.history[-2:], body.message])[-500:]
    if body.method == "images":
        result = image_search(q=query, limit=5, db=db)
    elif body.method == "keyword":
        from app.api.keyword_search import keyword_search
        result = keyword_search(q=query, limit=5, db=db)
    elif body.method == "semantic":
        from app.main import semantic_search
        result = semantic_search(q=query, limit=5, db=db)
    else:
        from app.api.hybrid_search import hybrid_search
        result = hybrid_search(q=query, limit=5, db=db)
    return {"query": query, "message": f"Found {len(result['results'])} sources for this search. Refine your next message to narrow the results.",
            "response_type": "retrieval_refinement", "results": result["results"]}


@router.post("/figures/{figure_id}/ocr")
def extract_figure_text(figure_id: int, retry: bool = False, db: Session = Depends(get_db)):
    # Serialize requests for this figure so double clicks cannot duplicate passages.
    figure = db.scalar(select(Figure).where(Figure.id == figure_id).with_for_update())
    if figure is None:
        raise HTTPException(404, "Figure not found")
    source = db.get(SourceAsset, figure.document_id)
    path = data_path(source.storage_path) if source and source.media_type.startswith("image/") else data_path(figure.storage_path)
    if not path.is_file():
        raise HTTPException(404, "Image file unavailable")
    try:
        with Image.open(path) as image:
            record = index_image_text(figure, image, db, embed_texts, retry)
        db.commit()
        return ocr_payload(record)
    except Exception:
        db.rollback()
        raise


@router.get("/documents/{document_id}/ocr")
def extracted_document_text(document_id: int, db: Session = Depends(get_db)):
    if db.get(Document, document_id) is None:
        raise HTTPException(404, "Document not found")
    records = db.scalars(select(ImageText).join(Figure, Figure.id == ImageText.figure_id)
                         .where(Figure.document_id == document_id).order_by(Figure.page_number, Figure.id)).all()
    return {"document_id": document_id, "results": [ocr_payload(record) for record in records]}
