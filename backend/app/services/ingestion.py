"""Turn PDFs, plain text and images into documents, embedded passages and figures.

Each ingest is one transaction: on any failure the database is rolled back and
the files written so far are deleted.
"""
import io
import logging
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import EMBED_BATCH_SIZE, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
from app.errors import UserError
from app.models import Document, ImageText, Passage
from app.services import embeddings
from app.services.assets import add_source, extract_figures, owned_files, store_figures
from app.services.ocr import extract_image_text

logger = logging.getLogger(__name__)


def extract_pdf_pages(pdf_path) -> list[dict]:
    """Extract text per page, using Tesseract OCR for pages with little or no selectable text."""
    pages = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < 40:
                try:
                    ocr_page = page.get_textpage_ocr(language="eng", dpi=200, full=True)
                    ocr_text = page.get_text("text", textpage=ocr_page).strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception as exc:  # Keep any selectable text if OCR is unavailable.
                    logger.warning("OCR failed on page %s of %s: %s", page_number, Path(pdf_path).name, exc)
            text = text.replace("\x00", "").strip()
            if text:
                pages.append({"page_number": page_number, "text": text})
    return pages


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into word-aligned chunks of about chunk_size characters, overlapping by up to overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")
    words = text.replace("\x00", "").split()
    chunks, start = [], 0
    while start < len(words):
        end, length = start, 0
        while end < len(words):
            extra = len(words[end]) + (1 if end > start else 0)
            if length + extra > chunk_size and end > start:
                break
            length += extra
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # Step back over trailing words that fit in the overlap budget.
        next_start, kept = end, 0
        while next_start > start:
            extra = len(words[next_start - 1]) + (1 if kept else 0)
            if kept + extra > overlap:
                break
            kept += extra
            next_start -= 1
        start = next_start if next_start > start else end
    return chunks


def add_passages(db, document_id, items) -> int:
    """Embed (content, page_number) pairs in batches and add them as passages."""
    for start in range(0, len(items), EMBED_BATCH_SIZE):
        batch = items[start:start + EMBED_BATCH_SIZE]
        vectors = embeddings.embed_texts([content for content, _ in batch])
        for (content, page), vector in zip(batch, vectors, strict=True):
            db.add(Passage(document_id=document_id, content=content, page_number=page, embedding=vector))
    return len(items)


def _new_document(db, filename, title, status="processing") -> Document:
    document = Document(filename=filename, title=title, status=status)
    db.add(document)
    db.flush()
    return document


def ingest_pdf(pdf_path, filename, db) -> dict:
    pages = extract_pdf_pages(pdf_path)
    figure_items = extract_figures(pdf_path)
    if not pages and not figure_items:
        raise UserError("No text could be extracted, even with OCR. "
                        "The PDF may be blank, damaged, or contain unreadable scans.")
    passages = [(chunk, page["page_number"]) for page in pages for chunk in split_text(page["text"])]
    with owned_files(db) as created:
        document = _new_document(db, filename, Path(filename).stem)
        add_source(db, created, document.id, Path(pdf_path), "application/pdf")
        figures = store_figures(figure_items, document.id, db, created)
        add_passages(db, document.id, passages)
        document.status = "processed"
        db.commit()
    return {"document_id": document.id, "filename": document.filename, "pages_with_text": len(pages),
            "passages_created": len(passages), "figures_created": len(figures),
            "embeddings_created": len(passages), "status": document.status}


def ingest_text(db, title, content, source_url=None, license=None) -> dict:
    chunks = split_text(content)
    if not chunks:
        raise UserError("Text cannot be blank")
    with owned_files(db) as created:
        document = _new_document(db, Path(title).name[:200] + ".txt", title)
        add_source(db, created, document.id, content.encode("utf-8"), "text/plain", source_url, license)
        add_passages(db, document.id, [(chunk, 1) for chunk in chunks])
        document.status = "processed"
        db.commit()
    return {"document_id": document.id, "passages_created": len(chunks), "status": document.status}


def load_image(data: bytes) -> tuple[Image.Image, bytes]:
    """Validate an uploaded image; return it (EXIF-oriented, full resolution) plus a PNG copy of it."""
    if len(data) > MAX_IMAGE_BYTES:
        raise UserError("Image exceeds 20 MB", 413)
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in {"PNG", "JPEG", "WEBP"}:
            raise UserError("Use a PNG, JPEG or WebP image")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise UserError("Images must be at most 20 megapixels", 413)
        image.load()
        image = ImageOps.exif_transpose(image)
        png = io.BytesIO()
        image.save(png, format="PNG")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise UserError("Image cannot be decoded safely") from exc
    return image, png.getvalue()


def ingest_image(db, filename, image, png: bytes) -> dict:
    """Index an image for CLIP search and OCR its full-resolution original into passages."""
    preview = image.convert("RGB")
    preview.thumbnail((1600, 1600))
    with owned_files(db) as created:
        document = _new_document(db, filename, Path(filename).stem, status="processed")
        add_source(db, created, document.id, png, "image/png")
        [figure] = store_figures([(1, preview, filename)], document.id, db, created)
        db.flush()
        record, passages = index_image_text(figure, image, db)
        db.commit()
    return {"document_id": document.id, "figures_created": 1, "passages_created": passages,
            "ocr": ocr_payload(record), "status": "processed"}


def index_image_text(figure, image, db, retry=False) -> tuple[ImageText, int]:
    """OCR an image and index its text as passages, at most once per figure.

    Extracted/no-text outcomes are cached; failures are retried only when retry=True.
    Returns the OCR record and the number of passages added.
    """
    record = db.get(ImageText, figure.id)
    if record and (not retry or record.status in {"extracted", "no_text"}):
        return record, 0
    result = extract_image_text(image)
    if record is None:
        record = ImageText(figure_id=figure.id)
        db.add(record)
    record.text, record.status, record.message = result.text, result.status, result.message
    chunks = split_text(result.text) if result.status == "extracted" else []
    return record, add_passages(db, figure.document_id, [(chunk, figure.page_number) for chunk in chunks])


def ocr_payload(record: ImageText) -> dict:
    return {"figure_id": record.figure_id, "text": record.text, "status": record.status, "message": record.message}
