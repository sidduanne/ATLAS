from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models.document import Passage
from app.services.embeddings import embed_query

router = APIRouter(tags=["Hybrid Search"])

RRF_K = 60


@router.get("/api/v1/search/hybrid")
def hybrid_search(
    q: str = Query(..., min_length=3, max_length=500),
    limit: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Combine keyword and semantic search with Reciprocal Rank Fusion."""
    from app.api.keyword_search import keyword_search

    query = q.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Search query cannot be empty.")

    candidate_limit = min(limit * 4, 20)

    try:
        # Semantic retrieval
        query_vector = embed_query(query)
        semantic_rows = (
            db.query(Passage)
            .filter(Passage.embedding.is_not(None))
            .order_by(Passage.embedding.cosine_distance(query_vector))
            .limit(candidate_limit)
            .all()
        )

        # Keyword retrieval. Supply the actual DB session explicitly.
        keyword_response = keyword_search(
            q=query,
            limit=candidate_limit,
            db=db,
        )
        keyword_results = keyword_response["results"]

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid retrieval failed: {type(exc).__name__}",
        ) from exc

    scores = defaultdict(float)
    passages = {}

    # Add one reciprocal-rank contribution from each result list.
    for rank, passage in enumerate(semantic_rows, start=1):
        scores[passage.id] += 1.0 / (RRF_K + rank)
        passages[passage.id] = passage

    for rank, result in enumerate(keyword_results, start=1):
        passage_id = result["passage_id"]
        scores[passage_id] += 1.0 / (RRF_K + rank)

        # Load keyword-only passages into the same result collection.
        if passage_id not in passages:
            passage = db.get(Passage, passage_id)
            if passage is not None:
                passages[passage_id] = passage

    ranked_ids = sorted(
        (pid for pid in scores if pid in passages),
        key=lambda pid: scores[pid],
        reverse=True,
    )

    results = []
    for passage_id in ranked_ids[:limit]:
        passage = passages[passage_id]
        document = passage.document

        results.append({
            "passage_id": passage.id,
            "document_id": passage.document_id,
            "filename": document.filename,
            "title": document.title,
            "page_number": passage.page_number,
            "content": passage.content,
            "rrf_score": round(scores[passage_id], 8),
        })

    return {
        "query": query,
        "method": "reciprocal_rank_fusion",
        "rrf_k": RRF_K,
        "results_count": len(results),
        "results": results,
    }
