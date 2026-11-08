
import logging
import os
from time import perf_counter
import tempfile
from pathlib import Path

import pymupdf
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.models.document import Document, Passage
from app.models.review import SourceAsset
from app.api.workspace import router as workspace_router, LATENCIES
from fastapi.middleware.cors import CORSMiddleware
from app.services.embeddings import embed_query
from app.services.ingestion import ingest_pdf

# Preserve existing search routers.
from app.api.keyword_search import router as keyword_search_router
from app.api.hybrid_search import router as hybrid_search_router

from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    docs_url=None,  # Use the custom Swagger page defined below.
    title="Multimodal Document Retrieval Engine",
    description="Search technical documents, passages, and images.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","), allow_methods=["GET", "POST", "PATCH"], allow_headers=["Content-Type"])
app.include_router(workspace_router)
from app.api.research import router as research_router
app.include_router(research_router)


@app.middleware("http")
async def request_timing(request, call_next):
    started = perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api/v1/search") or request.url.path == "/api/v1/chat":
        elapsed = (perf_counter() - started) * 1000
        LATENCIES.append((request.url.path, elapsed, response.status_code))
        response.headers["Server-Timing"] = f"search;dur={elapsed:.2f}"
    return response


# Application-level upload limits.
MAX_PDF_SIZE = 500 * 1024 * 1024       # 500 MB per PDF
MAX_BULK_TOTAL_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB per batch
MAX_BULK_FILES = 50


@app.get("/")
def root():
    return {
        "message": "Multimodal Document Retrieval Engine is running!",
        "status": "ok",
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(select(SourceAsset.document_id).limit(1))
        return {"status": "healthy", "service": "backend"}
    except Exception as exc:
        raise HTTPException(503, "Database unavailable or schema not initialized; run backend/init_db.py") from exc


@app.get("/api/v1/info")
def project_info():
    return {
        "project": "Multimodal Document Retrieval Engine",
        "version": "1.0.0",
        "features": [
            "PDF ingestion",
            "Bulk PDF upload",
            "OCR-supported scanned PDF ingestion",
            "Passage storage",
            "Semantic text search",
            "Keyword search",
            "Hybrid search",
            "CLIP image retrieval",
            "Image OCR with searchable extracted text",
            "Uncertainty-guided review",
            "Frozen-embedding classifier updates",
            "Source previews and evaluation reports",
        ],
        "upload_limits": {
            "max_pdf_size_mb": 500,
            "max_bulk_total_size_gb": 2,
            "max_files_per_batch": MAX_BULK_FILES,
        },
    }


def process_uploaded_pdf(file: UploadFile, db: Session) -> dict:
    """
    Validate and ingest one PDF.
    OCR is handled by app.services.ingestion.extract_pdf_pages.
    """
    filename = Path(file.filename or "").name

    if not filename or not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0

            while True:
                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_PDF_SIZE:
                    raise OverflowError(
                        "PDF exceeds the 500 MB per-file limit."
                    )

                temp_file.write(chunk)

        if size == 0:
            raise ValueError("The uploaded PDF is empty.")

        try:
            with pymupdf.open(str(temp_path)) as pdf:
                if not pdf.is_pdf:
                    raise ValueError(
                        "The uploaded file is not a valid PDF."
                    )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                "The PDF could not be opened or is malformed."
            ) from exc

        return ingest_pdf(temp_path, filename, db)

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/api/v1/documents/upload")
def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and ingest one PDF."""
    try:
        return process_uploaded_pdf(file, db)

    except OverflowError as exc:
        db.rollback()
        raise HTTPException(
            status_code=413,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        db.rollback()
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()
        logger.exception("PDF ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="PDF ingestion failed. Check the backend logs.",
        ) from exc

    finally:
        file.file.close()



@app.post("/api/v1/documents/bulk-upload")
def bulk_upload_pdfs(
    files: list[UploadFile] = File(
        ...,
        description="Select multiple PDF files",
    ),
    db: Session = Depends(get_db),
):
    """
    Upload multiple PDFs.
    Each file is processed independently and receives its own result.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one PDF.",
        )

    if len(files) > MAX_BULK_FILES:
        for uploaded_file in files:
            uploaded_file.file.close()

        raise HTTPException(
            status_code=413,
            detail=f"A batch can contain at most {MAX_BULK_FILES} files.",
        )

    results = []
    total_bytes = 0

    try:
        for uploaded_file in files:
            filename = Path(uploaded_file.filename or "").name

            try:
                # Determine the size of this file before processing it.
                uploaded_file.file.seek(0, 2)
                file_size = uploaded_file.file.tell()
                uploaded_file.file.seek(0)

                total_bytes += file_size

                if total_bytes > MAX_BULK_TOTAL_SIZE:
                    results.append({
                        "filename": filename,
                        "status": "failed",
                        "error": "The batch exceeds the 2 GB total limit.",
                    })
                    continue

                if file_size > MAX_PDF_SIZE:
                    results.append({
                        "filename": filename,
                        "status": "failed",
                        "error": "PDF exceeds the 500 MB per-file limit.",
                    })
                    continue

                result = process_uploaded_pdf(uploaded_file, db)

                results.append({
                    **result,
                    "status": "success",
                })

            except OverflowError as exc:
                db.rollback()
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": str(exc),
                })

            except ValueError as exc:
                db.rollback()
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": str(exc),
                })

            except Exception:
                db.rollback()
                logger.exception(
                    "Bulk ingestion failed for %s",
                    filename,
                )
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": "Ingestion failed. Check the backend logs.",
                })

            finally:
                uploaded_file.file.close()

        succeeded = sum(
            1 for item in results if item["status"] == "success"
        )

        return {
            "total_files": len(files),
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
            "results": results,
        }

    finally:
        for uploaded_file in files:
            if not uploaded_file.file.closed:
                uploaded_file.file.close()


@app.get("/api/v1/documents")
def list_documents(db: Session = Depends(get_db)):
    documents = db.scalars(
        select(Document).order_by(Document.id.desc())
    ).all()

    sources = {s.document_id: s for s in db.scalars(select(SourceAsset))}
    return [
        {
            "has_source": doc.id in sources,
            "source_url": sources[doc.id].source_url if doc.id in sources else None,
            "license": sources[doc.id].license if doc.id in sources else None,
            "id": doc.id,
            "filename": doc.filename,
            "title": doc.title,
            "status": doc.status,
            "created_at": doc.created_at,
        }
        for doc in documents
    ]


@app.get("/api/v1/documents/{document_id}/passages")
def list_passages(
    document_id: int,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    passages = db.scalars(
        select(Passage)
        .where(Passage.document_id == document_id)
        .order_by(Passage.id)
    ).all()

    return [
        {
            "id": passage.id,
            "page_number": passage.page_number,
            "content": passage.content,
        }
        for passage in passages
    ]


@app.get("/api/v1/search")
def semantic_search(
    q: str = Query(min_length=2, max_length=1000),
    limit: int = Query(default=5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Search passages using vector similarity."""
    if not q.strip():
        raise HTTPException(422, "Search query cannot be empty")
    query_vector = embed_query(q)

    distance = Passage.embedding.cosine_distance(query_vector)

    rows = db.execute(
        select(Passage, Document, distance.label("distance"))
        .join(Document, Passage.document_id == Document.id)
        .where(Passage.embedding.is_not(None))
        .order_by(distance.asc(), Passage.id.asc())
        .limit(limit)
    ).all()

    results = []

    for passage, document, distance_value in rows:
        results.append({
            "passage_id": passage.id,
            "document_id": document.id,
            "filename": document.filename,
            "title": document.title,
            "page_number": passage.page_number,
            "content": passage.content,
            "distance": float(distance_value),
            "similarity": 1.0 - float(distance_value),
        })

    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


# Register existing search routes.
app.include_router(keyword_search_router)
app.include_router(hybrid_search_router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    operation = (
        schema.get("paths", {})
        .get("/api/v1/documents/bulk-upload", {})
        .get("post", {})
    )

    request_body = operation.get("requestBody", {})
    content = request_body.get("content", {})

    multipart = content.get("multipart/form-data")

    if multipart:
        schema_ref = multipart.get("schema", {}).get("$ref")

        if schema_ref:
            schema_name = schema_ref.rsplit("/", 1)[-1]
            body_schema = (
                schema.get("components", {})
                .get("schemas", {})
                .get(schema_name)
            )

            if body_schema is not None:
                body_schema.setdefault("properties", {})["files"] = {
                    "type": "array",
                    "title": "Files",
                    "description": "Select multiple PDF files",
                    "items": {
                        "type": "string",
                        "format": "binary",
                    },
                }
                body_schema.setdefault("required", [])
                if "files" not in body_schema["required"]:
                    body_schema["required"].append("files")

        else:
            multipart["schema"] = {
                "type": "object",
                "required": ["files"],
                "properties": {
                    "files": {
                        "type": "array",
                        "title": "Files",
                        "description": "Select multiple PDF files",
                        "items": {
                            "type": "string",
                            "format": "binary",
                        },
                    }
                },
            }

    app.openapi_schema = schema
    return app.openapi_schema




app.openapi = custom_openapi


@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Bulk PDF Upload</title>
      <style>
        body {
          font-family: system-ui, sans-serif;
          max-width: 760px;
          margin: 50px auto;
          padding: 24px;
          color: #202124;
          background: #f7f8fa;
        }
        main {
          background: white;
          padding: 28px;
          border-radius: 16px;
          box-shadow: 0 4px 24px #0000000d;
        }
        input, button { font: inherit; }
        input { display: block; margin: 22px 0; }
        button {
          padding: 12px 22px;
          border: 0;
          border-radius: 8px;
          background: #2563eb;
          color: white;
          cursor: pointer;
        }
        button:disabled { opacity: .5; cursor: wait; }
        #files { line-height: 1.8; overflow-wrap: anywhere; }
        #status { margin-top: 20px; white-space: pre-wrap; }
      </style>
    </head>
    <body>
      <main>
        <h1>Bulk PDF Upload</h1>
        <p>Select multiple PDFs, then upload them all in one request.</p>

        <form id="uploadForm">
          <input
            id="pdfFiles"
            name="files"
            type="file"
            accept=".pdf,application/pdf"
            multiple
            required
          >
          <div id="files">No files selected.</div>
          <p id="size"></p>
          <button id="uploadButton" type="submit">Upload All</button>
        </form>

        <pre id="status"></pre>
      </main>

      <script>
        const picker = document.getElementById("pdfFiles");
        const fileList = document.getElementById("files");
        const sizeLabel = document.getElementById("size");
        const status = document.getElementById("status");
        const button = document.getElementById("uploadButton");

        picker.addEventListener("change", () => {
          const files = Array.from(picker.files);
          fileList.textContent = files.length
            ? files.map((f, i) => `${i + 1}. ${f.name}`).join("\\n")
            : "No files selected.";

          const total = files.reduce((sum, f) => sum + f.size, 0);
          sizeLabel.textContent =
            `${files.length} file(s), ${(total / 1024 / 1024).toFixed(1)} MB total`;
        });

        document.getElementById("uploadForm").addEventListener(
          "submit", async (event) => {
            event.preventDefault();

            const files = Array.from(picker.files);
            if (!files.length) return;

            const data = new FormData();
            files.forEach(file => data.append("files", file));

            button.disabled = true;
            status.textContent = "Uploading and processing PDFs...";

            try {
              const response = await fetch(
                "/api/v1/documents/bulk-upload",
                { method: "POST", body: data }
              );

              const result = await response.json();
              status.textContent = JSON.stringify(result, null, 2);

              if (!response.ok) {
                status.textContent =
                  `Request failed (${response.status}):\\n` +
                  JSON.stringify(result, null, 2);
              }
            } catch (error) {
              status.textContent = `Upload error: ${error.message}`;
            } finally {
              button.disabled = false;
            }
          }
        );
      </script>
    </body>
    </html>
    """


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def custom_docs():
    swagger_html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
    ).body.decode("utf-8")

    upload_panel = """
    <div style="
        margin: 24px 3%;
        padding: 20px;
        border: 1px solid #49cc90;
        border-radius: 8px;
        font-family: Arial, sans-serif;
    ">
      <h2>Bulk Upload PDFs</h2>
      <p>Select multiple PDFs in one file dialog, then upload them together.</p>

      <input
        id="bulk-pdf-picker"
        type="file"
        accept=".pdf,application/pdf"
        multiple
      />

      <button
        id="bulk-upload-button"
        type="button"
        style="padding: 9px 18px; margin-left: 10px; cursor: pointer;"
      >Upload All</button>

      <pre id="bulk-upload-status"
           style="white-space: pre-wrap; overflow-wrap: anywhere;"></pre>
    </div>
    """

    upload_script = """
    <script>
    (() => {
      const button = document.getElementById("bulk-upload-button");
      const picker = document.getElementById("bulk-pdf-picker");
      const status = document.getElementById("bulk-upload-status");

      button.addEventListener("click", async () => {
        const files = Array.from(picker.files || []);

        if (!files.length) {
          status.textContent = "Select one or more PDF files first.";
          return;
        }

        const invalid = files.filter(
          file => !file.name.toLowerCase().endsWith(".pdf")
        );

        if (invalid.length) {
          status.textContent = "Only PDF files are allowed.";
          return;
        }

        const data = new FormData();
        files.forEach(file => data.append("files", file));

        button.disabled = true;
        status.textContent = `Uploading ${files.length} PDF(s)...`;

        try {
          const response = await fetch(
            "/api/v1/documents/bulk-upload",
            { method: "POST", body: data }
          );

          const result = await response.json();
          status.textContent =
            `HTTP ${response.status}\\n` +
            JSON.stringify(result, null, 2);
        } catch (error) {
          status.textContent = "Upload failed: " + error.message;
        } finally {
          button.disabled = false;
        }
      });
    })();
    </script>
    """

    swagger_html = swagger_html.replace(
        '<div id="swagger-ui">',
        upload_panel + '<div id="swagger-ui">',
        1,
    )
    swagger_html = swagger_html.replace("</body>", upload_script + "</body>", 1)

    return HTMLResponse(content=swagger_html)
