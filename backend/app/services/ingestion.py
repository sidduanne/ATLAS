
from pathlib import Path

import pymupdf
from sqlalchemy.orm import Session

from app.models.document import Document, Passage
from app.services.embeddings import embed_texts
from app.services.assets import extract_figures, save_source, store_figures
from app.config import data_path


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text page by page.
    Use Tesseract OCR when a page has little or no selectable text.
    """
    pages = []

    with pymupdf.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # Scanned pages usually have no selectable text.
            # Also OCR pages with only a tiny amount of extracted text.
            if len(text) < 40:
                try:
                    text_page = page.get_textpage_ocr(
                        language="eng",
                        dpi=200,
                        full=True,
                    )
                    ocr_text = page.get_text(
                        "text",
                        textpage=text_page,
                    ).strip()

                    if len(ocr_text) > len(text):
                        text = ocr_text
                except Exception as exc:
                    # Preserve any normal text if OCR fails on a page.
                    print(
                        f"OCR failed on page {page_number} "
                        f"of {Path(pdf_path).name}: {exc}"
                    )

            text = text.replace("\x00", "").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text,
                })

    return pages


def split_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """Split text into overlapping chunks, measured approximately in characters."""
    text = text.replace("\x00", "")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start
        length = 0

        while end < len(words):
            word_length = len(words[end])
            additional_length = word_length + (1 if end > start else 0)

            if length + additional_length > chunk_size and end > start:
                break

            length += additional_length
            end += 1

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        next_start = end
        retained_length = 0

        while next_start > start:
            word_length = len(words[next_start - 1])
            additional_length = word_length + (1 if retained_length else 0)

            if retained_length + additional_length > overlap:
                break

            retained_length += additional_length
            next_start -= 1

        start = next_start if next_start > start else end

    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    filename: str,
    db: Session,
) -> dict:
    """Extract, chunk, embed, and store a PDF in a single transaction."""
    pages = extract_pdf_pages(pdf_path)

    figure_items = extract_figures(pdf_path, 0)

    if not pages and not figure_items:
        raise ValueError(
            "No text could be extracted, even with OCR. "
            "The PDF may be blank, damaged, or contain unreadable scans."
        )

    passage_items = []

    for page in pages:
        for chunk in split_text(page["text"]):
            passage_items.append({
                "content": chunk,
                "page_number": page["page_number"],
            })


    document = Document(
        filename=filename,
        title=Path(filename).stem,
        status="processing",
    )

    created_paths = []
    try:
        db.add(document)
        db.flush()

        source = save_source(pdf_path, document.id, "application/pdf")
        created_paths.append(data_path(source.storage_path))
        db.add(source)
        figures_created = store_figures(figure_items, document.id, db, created_paths)

        # Embed in batches to limit memory usage.
        for start in range(0, len(passage_items), 32):
            batch = passage_items[start:start + 32]
            vectors = embed_texts([item["content"] for item in batch])

            for item, vector in zip(batch, vectors, strict=True):
                db.add(
                    Passage(
                        document_id=document.id,
                        content=item["content"],
                        page_number=item["page_number"],
                        embedding=vector,
                    )
                )

        document.status = "processed"
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise

    return {
        "document_id": document.id,
        "filename": document.filename,
        "pages_with_text": len(pages),
        "passages_created": len(passage_items),
        "figures_created": figures_created,
        "embeddings_created": len(passage_items),
        "status": document.status,
    }
