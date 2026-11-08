"""Library endpoints: uploads, source files, provenance, figures and image text."""
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field, HttpUrl
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import MAX_BATCH_BYTES, MAX_BATCH_FILES, MAX_IMAGE_BYTES, MAX_PDF_BYTES, MB, data_path
from app.dependencies import get_db
from app.errors import UserError
from app.models import Document, Figure, ImageText, Passage, SourceAsset
from app.services.assets import attach_pdf, owned_files
from app.services.ingestion import index_image_text, ingest_image, ingest_pdf, ingest_text, load_image, ocr_payload
from app.services.search import item_payload

router = APIRouter(prefix="/api/v1", tags=["Documents"])
logger = logging.getLogger(__name__)


def get_or_404(db, model, item_id, name):
    item = db.get(model, item_id)
    if item is None:
        raise HTTPException(404, f"{name} not found.")
    return item


@contextmanager
def spooled_pdf(file: UploadFile):
    """Copy an uploaded PDF to a temporary file (size-limited) and check that it opens."""
    with tempfile.TemporaryDirectory() as directory:
        path, size = Path(directory) / "upload.pdf", 0
        with path.open("wb") as output:
            while chunk := file.file.read(MB):
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    raise UserError("PDF exceeds the 500 MB per-file limit.", 413)
                output.write(chunk)
        if size == 0:
            raise UserError("The uploaded PDF is empty.")
        try:
            with pymupdf.open(path) as pdf:
                valid = pdf.is_pdf and not pdf.needs_pass
        except Exception as exc:
            raise UserError("The PDF could not be opened or is malformed.") from exc
        if not valid:
            raise UserError("Upload a valid PDF that does not require a password.")
        yield path


def ingest_upload(file: UploadFile, db) -> dict:
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise UserError("Only PDF files are supported.")
    with spooled_pdf(file) as path:
        return ingest_pdf(path, filename, db)


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    sources = {source.document_id: source for source in db.scalars(select(SourceAsset))}
    return [{"id": doc.id, "filename": doc.filename, "title": doc.title, "status": doc.status,
             "created_at": doc.created_at, "has_source": doc.id in sources,
             "source_url": sources[doc.id].source_url if doc.id in sources else None,
             "license": sources[doc.id].license if doc.id in sources else None}
            for doc in db.scalars(select(Document).order_by(Document.id.desc()))]


@router.get("/documents/{document_id}/passages")
def list_passages(document_id: int, db: Session = Depends(get_db)):
    get_or_404(db, Document, document_id, "Document")
    passages = db.scalars(select(Passage).where(Passage.document_id == document_id).order_by(Passage.id))
    return [{"id": p.id, "page_number": p.page_number, "content": p.content} for p in passages]


@router.post("/documents/upload")
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        return ingest_upload(file, db)
    except UserError:
        raise
    except Exception as exc:
        logger.exception("PDF ingestion failed")
        raise HTTPException(500, "PDF ingestion failed. Check the backend logs.") from exc


@router.post("/documents/bulk-upload")
def bulk_upload(files: list[UploadFile] = File(..., description="Select multiple PDF files"),
                db: Session = Depends(get_db)):
    """Ingest each PDF independently; one failure does not affect the others."""
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(413, f"A batch can contain at most {MAX_BATCH_FILES} files.")
    results, total = [], 0
    for file in files:
        filename = Path(file.filename or "").name
        file.file.seek(0, 2)
        total += file.file.tell()
        file.file.seek(0)
        try:
            if total > MAX_BATCH_BYTES:
                raise UserError("The batch exceeds the 2 GB total limit.")
            results.append({**ingest_upload(file, db), "status": "success"})
        except UserError as exc:
            results.append({"filename": filename, "status": "failed", "error": str(exc)})
        except Exception:
            logger.exception("Bulk ingestion failed for %s", filename)
            results.append({"filename": filename, "status": "failed", "error": "Ingestion failed. Check the backend logs."})
    succeeded = sum(result["status"] == "success" for result in results)
    return {"total_files": len(files), "succeeded": succeeded, "failed": len(results) - succeeded, "results": results}


class TextDocument(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=2, max_length=1_000_000)
    source_url: HttpUrl | None = None
    license: str | None = Field(None, max_length=200)


@router.post("/documents/upload-text")
def upload_text(body: TextDocument, db: Session = Depends(get_db)):
    return ingest_text(db, body.title, body.content, str(body.source_url) if body.source_url else None, body.license)


@router.post("/documents/upload-image")
def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    image, png = load_image(file.file.read(MAX_IMAGE_BYTES + 1))
    return ingest_image(db, Path(file.filename or "image.png").name[:255], image, png)


@router.get("/documents/{document_id}/source")
def document_source(document_id: int, db: Session = Depends(get_db)):
    source = db.get(SourceAsset, document_id)
    if not source or not data_path(source.storage_path).is_file():
        raise HTTPException(404, "Source unavailable. Attach the original file to this legacy record.")
    return FileResponse(data_path(source.storage_path), media_type=source.media_type,
                        headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})


@router.post("/documents/{document_id}/source")
def attach_source(document_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    document = get_or_404(db, Document, document_id, "Document")
    if Path(file.filename or "").name != document.filename or not document.filename.lower().endswith(".pdf"):
        raise UserError("Select the original PDF with the same filename as this document")
    with spooled_pdf(file) as path:
        return attach_pdf(path, document, db)


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


@router.get("/documents/{document_id}/figures")
def document_figures(document_id: int, db: Session = Depends(get_db)):
    document = get_or_404(db, Document, document_id, "Document")
    figures = db.scalars(select(Figure).where(Figure.document_id == document_id).order_by(Figure.id))
    return [item_payload(figure, "figure", document) for figure in figures]


@router.get("/figures/{figure_id}/image")
def figure_image(figure_id: int, db: Session = Depends(get_db)):
    path = data_path(get_or_404(db, Figure, figure_id, "Figure").storage_path)
    if not path.is_file():
        raise HTTPException(404, "Figure file unavailable")
    return FileResponse(path, media_type="image/png")


@router.post("/figures/{figure_id}/ocr")
def extract_figure_text(figure_id: int, retry: bool = False, db: Session = Depends(get_db)):
    # Lock the figure so double clicks cannot index the same text twice.
    figure = db.scalar(select(Figure).where(Figure.id == figure_id).with_for_update())
    if figure is None:
        raise HTTPException(404, "Figure not found.")
    source = db.get(SourceAsset, figure.document_id)
    # Uploaded images are read from the full-resolution original; PDF figures from their crop.
    is_image = source is not None and source.media_type.startswith("image/")
    path = data_path(source.storage_path if is_image else figure.storage_path)
    if not path.is_file():
        raise HTTPException(404, "Image file unavailable")
    with owned_files(db), Image.open(path) as image:
        record, _ = index_image_text(figure, image, db, retry)
        db.commit()
    return ocr_payload(record)


@router.get("/documents/{document_id}/ocr")
def document_image_text(document_id: int, db: Session = Depends(get_db)):
    get_or_404(db, Document, document_id, "Document")
    records = db.scalars(select(ImageText).join(Figure, Figure.id == ImageText.figure_id)
                         .where(Figure.document_id == document_id).order_by(Figure.page_number, Figure.id))
    return {"document_id": document_id, "results": [ocr_payload(record) for record in records]}
