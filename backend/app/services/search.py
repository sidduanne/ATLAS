"""Retrieval methods shared by the search API, conversation search and relevance pooling."""
import re
from collections import defaultdict, deque

from sqlalchemy import func, select

from app.errors import UserError
from app.models import Document, Figure, Passage
from app.services import clip, embeddings

# (path, milliseconds, status) of the last 1000 search/chat requests in this process.
LATENCIES: deque = deque(maxlen=1000)
RRF_K = 60
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "compared", "does", "for", "from", "how", "in",
              "is", "it", "of", "on", "or", "the", "to", "was", "what", "when", "where", "which", "with"}
SCORE_KEYS = ("distance", "similarity", "keyword_rank")


def item_payload(item, item_type, document) -> dict:
    """Common JSON shape for a passage or figure with its source document."""
    return {"item_id": item.id, "item_type": item_type, "document_id": item.document_id,
            "filename": document.filename, "page_number": item.page_number,
            "content": item.content if item_type == "passage" else item.caption,
            "preview_url": f"/api/v1/figures/{item.id}/image" if item_type == "figure" else None,
            "source_url": f"/api/v1/documents/{item.document_id}/source"}


def _passage_result(passage, document, **scores) -> dict:
    return {"passage_id": passage.id, "document_id": document.id, "filename": document.filename,
            "title": document.title, "page_number": passage.page_number, "content": passage.content, **scores}


def _clean(query: str) -> str:
    query = query.strip()
    if not query:
        raise UserError("Search query cannot be empty.")
    return query


def semantic(db, query, limit) -> list[dict]:
    """Sentence-BERT cosine similarity over passage embeddings."""
    distance = Passage.embedding.cosine_distance(embeddings.embed_query(_clean(query))).label("distance")
    rows = db.execute(select(Passage, Document, distance).join(Document, Passage.document_id == Document.id)
                      .where(Passage.embedding.is_not(None)).order_by(distance, Passage.id).limit(limit)).all()
    return [_passage_result(p, d, distance=float(dist), similarity=1.0 - float(dist)) for p, d, dist in rows]


def keyword(db, query, limit) -> list[dict]:
    """PostgreSQL full-text search; any meaningful query term may match (OR)."""
    terms = [word for word in re.findall(r"[a-zA-Z0-9]+", _clean(query).lower()) if word not in STOP_WORDS]
    if not terms:
        raise UserError("Query must contain at least one searchable term.")
    document_vector = func.to_tsvector("english", Passage.content)
    ts_query = func.to_tsquery("english", " | ".join(terms))
    rank = func.ts_rank_cd(document_vector, ts_query).label("rank")
    rows = db.execute(select(Passage, Document, rank).join(Document, Document.id == Passage.document_id)
                      .where(document_vector.op("@@")(ts_query)).order_by(rank.desc(), Passage.id).limit(limit)).all()
    return [_passage_result(p, d, keyword_rank=round(float(score), 6)) for p, d, score in rows]


def hybrid(db, query, limit) -> list[dict]:
    """Reciprocal-rank fusion of semantic and keyword candidates."""
    candidates = min(limit * 4, 20)
    scores, results = defaultdict(float), {}
    for ranked in (semantic(db, query, candidates), keyword(db, query, candidates)):
        for rank, result in enumerate(ranked, start=1):
            scores[result["passage_id"]] += 1.0 / (RRF_K + rank)
            results.setdefault(result["passage_id"], result)
    best = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [{**{k: v for k, v in results[pid].items() if k not in SCORE_KEYS}, "rrf_score": round(scores[pid], 8)}
            for pid in best]


def images(db, query, limit) -> list[dict]:
    """CLIP text-to-image similarity over figure embeddings."""
    distance = Figure.embedding.cosine_distance(clip.embed_image_query(_clean(query))).label("distance")
    rows = db.execute(select(Figure, Document, distance).join(Document, Figure.document_id == Document.id)
                      .where(Figure.embedding.is_not(None)).order_by(distance, Figure.id).limit(limit)).all()
    return [{**item_payload(f, "figure", d), "figure_id": f.id, "similarity": 1 - float(dist)} for f, d, dist in rows]


METHODS = {"semantic": semantic, "keyword": keyword, "hybrid": hybrid, "images": images}
