"""Turn PDFs, plain text and images into documents, embedded passages and figures.

Every ingest is one transaction: on failure the database is rolled back and the
files written so far are deleted. All writes stay inside DATA_DIR.
"""
import hashlib
import io
import logging
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core import EMBED_BATCH_SIZE, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, UserError, data_path
from app.models import Document, Figure, ImageText, Passage, SourceAsset
from app.services import encoders

logger = logging.getLogger(__name__)
SOURCE_SUFFIXES = {"application/pdf": ".pdf", "text/plain": ".txt"}


# --- Files ---------------------------------------------------------------------------------------

@contextmanager
def owned_files(db):
    """Yield a list for paths written inside the block; on failure roll back and delete them."""
    created: list[Path] = []
    try:
        yield created
    except BaseException:
        db.rollback()
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _write(relative: str, data: bytes, created: list[Path]) -> str:
    target = data_path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    created.append(target)
    target.write_bytes(data)
    return relative


def add_source(db, created, document_id, source: Path | bytes, media_type, source_url=None, license=None):
    """Store the original file (a path or raw bytes) as the document's source."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    relative = _write(f"sources/{document_id}/{uuid4().hex}{SOURCE_SUFFIXES.get(media_type, '.png')}", data, created)
    db.add(SourceAsset(document_id=document_id, storage_path=relative, sha256=hashlib.sha256(data).hexdigest(),
                       media_type=media_type, source_url=source_url, license=license))


# --- Text --------------------------------------------------------------------------------------

def extract_pdf_pages(pdf_path) -> list[dict]:
    """Extract text per page, using Tesseract OCR for pages with little or no selectable text."""
    pages = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if len(text) < 40:
                try:
                    ocr_text = page.get_text("text", textpage=page.get_textpage_ocr(language="eng", dpi=200, full=True)).strip()
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
        vectors = encoders.embed_texts([content for content, _ in batch])
        for (content, page), vector in zip(batch, vectors, strict=True):
            db.add(Passage(document_id=document_id, content=content, page_number=page, embedding=vector))
    return len(items)


# --- Figures -----------------------------------------------------------------------------------

def extract_figures(pdf_path):
    """Return (page, image, nearby text) for embedded raster images and vector drawing clusters."""
    items = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, 1):
            rects = []
            for info in page.get_image_info():
                rect = pymupdf.Rect(info["bbox"]) & page.rect
                if rect.width >= 32 and rect.height >= 32 and rect not in rects:
                    rects.append(rect)
            # Diagram-heavy scientific PDFs often contain vector drawings rather than images.
            for rect in page.cluster_drawings() if hasattr(page, "cluster_drawings") else []:
                rect = rect & page.rect
                if rect.width >= 60 and rect.height >= 40 and not any(rect.intersects(r) for r in rects):
                    rects.append(rect)
            for rect in rects:
                pixmap = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
                image.thumbnail((1600, 1600))
                caption = page.get_text("text", clip=rect + (-10, -10, 10, 60)).strip()[:1000]
                items.append((page_number, image, caption))
    return items


def store_figures(items, document_id, db, created_paths) -> list[Figure]:
    """CLIP-embed (page, image, caption) items in small batches and store them as figures."""
    figures = []
    for start in range(0, len(items), 8):
        batch = items[start:start + 8]
        vectors = encoders.embed_images([image for _, image, _ in batch])
        for (page, image, caption), vector in zip(batch, vectors, strict=True):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            relative = _write(f"figures/{document_id}/{uuid4().hex}.png", buffer.getvalue(), created_paths)
            figure = Figure(document_id=document_id, page_number=page, caption=caption.replace("\x00", ""),
                            storage_path=relative, embedding=vector)
            db.add(figure)
            figures.append(figure)
    return figures


# --- OCR ---------------------------------------------------------------------------------------

@dataclass
class OCRResult:
    text: str
    status: str  # extracted | no_text | unavailable | timeout | failed
    message: str | None = None


def extract_image_text(image: Image.Image) -> OCRResult:
    """Run local Tesseract on an image with a 60-second limit and an explicit outcome."""
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > MAX_IMAGE_PIXELS:
        return OCRResult("", "failed", "OCR supports images up to 20 megapixels")
    # Transparent screenshots need a white background, not a black alpha discard.
    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    command = [os.getenv("TESSERACT_CMD", "tesseract"), "stdin", "stdout", "-l", os.getenv("OCR_LANGUAGE", "eng"), "--psm", "3"]
    try:
        result = subprocess.run(command, input=buffer.getvalue(), capture_output=True, timeout=60, check=False)
    except FileNotFoundError:
        return OCRResult("", "unavailable", "Tesseract is not installed or TESSERACT_CMD is incorrect")
    except subprocess.TimeoutExpired:
        return OCRResult("", "timeout", "Text extraction exceeded 60 seconds; try a smaller or clearer image")
    except OSError:
        return OCRResult("", "failed", "Tesseract could not be started")
    if result.returncode:
        return OCRResult("", "failed", "Tesseract could not process the image; check OCR language data")
    text = result.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
    if not any(character.isalnum() for character in text):
        return OCRResult("", "no_text", "No readable text detected; the image is still available for visual search")
    return OCRResult(text, "extracted")


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


# --- Ingest entry points -----------------------------------------------------------------------

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


def attach_pdf(path, document: Document, db) -> dict:
    """Attach the original PDF to an existing (legacy) record and index its figures; passages are untouched."""
    if db.get(SourceAsset, document.id):
        raise UserError("This document already has a source. Upload it as a new document instead.")
    with owned_files(db) as created:
        add_source(db, created, document.id, Path(path), "application/pdf")
        figures = store_figures(extract_figures(path), document.id, db, created)
        db.commit()
    return {"document_id": document.id, "figures_created": len(figures)}


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
