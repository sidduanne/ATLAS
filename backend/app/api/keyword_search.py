
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models.document import Document, Passage

router = APIRouter()

# Common English words that contribute little to keyword matching.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "compared", "does", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "was", "what",
    "when", "where", "which", "with",
}


@router.get("/api/v1/search/keyword")
def keyword_search(
    q: str = Query(min_length=3, max_length=500),
    limit: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Find passages using PostgreSQL full-text keyword matching."""
    query = q.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Search query cannot be empty.",
        )

    # Extract safe word tokens before building a PostgreSQL tsquery.
    terms = [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if word not in STOP_WORDS
    ]

    if not terms:
        raise HTTPException(
            status_code=422,
            detail="Query must contain at least one searchable term.",
        )

    # OR matching gives a useful baseline for natural-language questions:
    # a passage can match any meaningful query term.
    tsquery_text = " | ".join(terms)

    search_vector = func.to_tsvector("english", Passage.content)
    search_query = func.to_tsquery("english", tsquery_text)

    rank = func.ts_rank_cd(
        search_vector,
        search_query,
    ).label("rank")

    statement = (
        select(Passage, Document, rank)
        .join(Document, Document.id == Passage.document_id)
        .where(search_vector.op("@@")(search_query))
        .order_by(rank.desc(), Passage.id.asc())
        .limit(limit)
    )

    rows = db.execute(statement).all()

    results = []

    for passage, document, score in rows:
        results.append({
            "passage_id": passage.id,
            "document_id": document.id,
            "filename": document.filename,
            "title": document.title,
            "page_number": passage.page_number,
            "content": passage.content,
            "keyword_rank": round(float(score), 6),
        })

    return {
        "query": query,
        "method": "postgresql_full_text_search",
        "results_count": len(results),
        "results": results,
    }
