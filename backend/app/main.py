"""FastAPI application: routers, error rendering and search latency telemetry."""
import logging
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import documents, research, review, search
from app.core import CORS_ORIGINS, MAX_BATCH_BYTES, MAX_BATCH_FILES, MAX_PDF_BYTES, MB, UserError, get_db
from app.models import SourceAsset
from app.services.search import LATENCIES

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Multimodal Document Retrieval Engine", version="1.0.0",
              description="Search technical documents, passages, and images.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST", "PATCH"],
                   allow_headers=["Content-Type"])
for module in (documents, search, review, research):
    app.include_router(module.router)


@app.exception_handler(UserError)
async def render_user_error(_: Request, exc: UserError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)


@app.middleware("http")
async def time_searches(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api/v1/search") or request.url.path == "/api/v1/chat":
        elapsed = (perf_counter() - started) * 1000
        LATENCIES.append((request.url.path, elapsed, response.status_code))
        response.headers["Server-Timing"] = f"search;dur={elapsed:.2f}"
    return response


@app.get("/")
def root():
    return {"message": "Multimodal Document Retrieval Engine is running!", "status": "ok"}


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(select(SourceAsset.document_id).limit(1))
    except Exception as exc:
        raise HTTPException(503, "Database unavailable or schema not initialized; run backend/init_db.py") from exc
    return {"status": "healthy", "service": "backend"}


@app.get("/api/v1/info")
def project_info():
    return {"project": app.title, "version": app.version,
            "features": ["PDF ingestion", "Bulk PDF upload", "OCR-supported scanned PDF ingestion", "Passage storage",
                         "Semantic text search", "Keyword search", "Hybrid search", "CLIP image retrieval",
                         "Image OCR with searchable extracted text", "Uncertainty-guided review",
                         "Frozen-embedding classifier updates", "Source previews and evaluation reports"],
            "upload_limits": {"max_pdf_size_mb": MAX_PDF_BYTES // MB, "max_bulk_total_size_gb": MAX_BATCH_BYTES // 1024 // MB,
                              "max_files_per_batch": MAX_BATCH_FILES}}
