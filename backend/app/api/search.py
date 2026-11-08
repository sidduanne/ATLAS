"""Search endpoints. The retrieval logic lives in app.services.search."""
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy.orm import Session

from app.core import get_db
from app.services import search

router = APIRouter(prefix="/api/v1", tags=["Search"])


@router.get("/search")
def semantic_search(q: str = Query(min_length=2, max_length=1000), limit: int = Query(5, ge=1, le=50),
                    db: Session = Depends(get_db)):
    results = search.semantic(db, q, limit)
    return {"query": q.strip(), "count": len(results), "results": results}


@router.get("/search/keyword")
def keyword_search(q: str = Query(min_length=3, max_length=500), limit: int = Query(5, ge=1, le=20),
                   db: Session = Depends(get_db)):
    results = search.keyword(db, q, limit)
    return {"query": q.strip(), "method": "postgresql_full_text_search", "results_count": len(results), "results": results}


@router.get("/search/hybrid")
def hybrid_search(q: str = Query(min_length=3, max_length=500), limit: int = Query(5, ge=1, le=20),
                  db: Session = Depends(get_db)):
    results = search.hybrid(db, q, limit)
    return {"query": q.strip(), "method": "reciprocal_rank_fusion", "rrf_k": search.RRF_K,
            "results_count": len(results), "results": results}


@router.get("/search/images")
def image_search(q: str = Query(min_length=2, max_length=500), limit: int = Query(5, ge=1, le=50),
                 db: Session = Depends(get_db)):
    return {"query": q.strip(), "method": "clip", "results": search.images(db, q, limit)}


Turn = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=500)]


class ChatBody(BaseModel):
    message: Turn
    history: list[Annotated[str, Field(max_length=500)]] = Field(default_factory=list, max_length=10)
    method: Literal["semantic", "keyword", "hybrid", "images"] = "hybrid"


@router.post("/chat")
def chat(body: ChatBody, db: Session = Depends(get_db)):
    """Conversational refinement: search with the last two user turns prepended. No generated answers."""
    query = " ".join([*body.history[-2:], body.message])[-500:]
    results = search.METHODS[body.method](db, query, 5)
    return {"query": query, "response_type": "retrieval_refinement", "results": results,
            "message": f"Found {len(results)} sources for this search. Refine your next message to narrow the results."}
