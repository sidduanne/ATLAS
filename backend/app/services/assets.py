"""Owned source files and figure extraction; all writes are confined to DATA_DIR."""
import hashlib
import io
import shutil
from pathlib import Path
from uuid import uuid4

import pymupdf
from PIL import Image
from sqlalchemy import select

from app.config import data_path
from app.models import Document, Figure, SourceAsset
from app.services.clip import embed_images


def save_source(path, document_id, media_type, source_url=None, license=None):
    suffix = {"application/pdf": ".pdf", "text/plain": ".txt"}.get(media_type, ".png")
    relative = f"sources/{document_id}/{uuid4().hex}{suffix}"
    target = data_path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(path, target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return SourceAsset(document_id=document_id, storage_path=relative,
                       sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                       media_type=media_type, source_url=source_url, license=license)


def extract_figures(pdf_path, document_id):
    """Extract embedded raster figures plus vector drawing clusters as rendered crops."""
    items = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, 1):
            rectangles = []
            for image in page.get_image_info():
                rect = pymupdf.Rect(image["bbox"]) & page.rect
                if rect.width >= 32 and rect.height >= 32 and rect not in rectangles:
                    rectangles.append(rect)
            # Diagram-heavy scientific PDFs often contain vector drawings, not images.
            if hasattr(page, "cluster_drawings"):
                for rect in page.cluster_drawings():
                    rect = rect & page.rect
                    if rect.width >= 60 and rect.height >= 40 and not any(rect.intersects(r) for r in rectangles):
                        rectangles.append(rect)
            for rect in rectangles:
                pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                image.thumbnail((1600, 1600))
                items.append((page_number, image, page.get_text("text", clip=rect + (-10, -10, 10, 60)).strip()[:1000]))
    return items


def store_figures(items, document_id, db, created_paths):
    count = 0
    for start in range(0, len(items), 8):
        batch = items[start:start + 8]
        vectors = embed_images([item[1] for item in batch])
        for (page, image, caption), vector in zip(batch, vectors, strict=True):
            relative = f"figures/{document_id}/{uuid4().hex}.png"
            target = data_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            created_paths.append(target)
            image.save(target)
            db.add(Figure(document_id=document_id, page_number=page, caption=caption.replace("\x00", ""),
                          storage_path=relative, embedding=vector))
            count += 1
    return count


def attach_pdf(path, document: Document, db):
    if db.get(SourceAsset, document.id):
        raise ValueError("This document already has a source. Upload it as a new document instead.")
    created = []
    try:
        source = save_source(path, document.id, "application/pdf")
        created.append(data_path(source.storage_path))
        db.add(source)
        count = store_figures(extract_figures(path, document.id), document.id, db, created)
        db.commit()
        return {"document_id": document.id, "figures_created": count}
    except Exception:
        db.rollback()
        for target in created:
            target.unlink(missing_ok=True)
        raise
