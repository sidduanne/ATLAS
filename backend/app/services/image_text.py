from app.models import ImageText, Passage
from app.services.ocr import extract_image_text


def index_image_text(figure, image, db, embedder, retry=False):
    """Cache OCR; never replace already indexed passages or their human labels."""
    from app.services.ingestion import split_text
    record = db.get(ImageText, figure.id)
    if record and (not retry or record.status in {"extracted", "no_text"}):
        return record
    result = extract_image_text(image)
    if record is None:
        record = ImageText(figure_id=figure.id)
        db.add(record)
    record.text, record.status, record.message = result.text, result.status, result.message
    chunks = split_text(result.text) if result.status == "extracted" else []
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start+32]
        for content, vector in zip(batch, embedder(batch), strict=True):
            db.add(Passage(document_id=figure.document_id, content=content,
                           page_number=figure.page_number, embedding=vector))
    return record


def ocr_payload(record):
    return {"figure_id": record.figure_id, "text": record.text, "status": record.status, "message": record.message}
