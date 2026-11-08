"""Owned source files and figure crops. Every write stays inside DATA_DIR."""
import hashlib
import io
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pymupdf
from PIL import Image

from app.config import data_path
from app.errors import UserError
from app.models import Document, Figure, SourceAsset
from app.services import clip

SOURCE_SUFFIXES = {"application/pdf": ".pdf", "text/plain": ".txt"}


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
    """Store the original file (a path or raw bytes) and register it as the document's source."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    relative = _write(f"sources/{document_id}/{uuid4().hex}{SOURCE_SUFFIXES.get(media_type, '.png')}", data, created)
    asset = SourceAsset(document_id=document_id, storage_path=relative, sha256=hashlib.sha256(data).hexdigest(),
                        media_type=media_type, source_url=source_url, license=license)
    db.add(asset)
    return asset


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
        vectors = clip.embed_images([image for _, image, _ in batch])
        for (page, image, caption), vector in zip(batch, vectors, strict=True):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            relative = _write(f"figures/{document_id}/{uuid4().hex}.png", buffer.getvalue(), created_paths)
            figure = Figure(document_id=document_id, page_number=page, caption=caption.replace("\x00", ""),
                            storage_path=relative, embedding=vector)
            db.add(figure)
            figures.append(figure)
    return figures


def attach_pdf(path, document: Document, db):
    """Attach the original PDF to an existing (legacy) record and index its figures; passages are untouched."""
    if db.get(SourceAsset, document.id):
        raise UserError("This document already has a source. Upload it as a new document instead.")
    with owned_files(db) as created:
        add_source(db, created, document.id, Path(path), "application/pdf")
        figures = store_figures(extract_figures(path), document.id, db, created)
        db.commit()
    return {"document_id": document.id, "figures_created": len(figures)}
