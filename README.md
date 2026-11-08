# Multimodal Document Retrieval & Active Learning Engine

A research prototype for technical PDF/text/image search, conversation-based search refinement, human anomaly tagging, and reproducible experiments. React + TypeScript, FastAPI, PostgreSQL + pgvector, frozen MiniLM text embeddings, and frozen CLIP image/text embeddings.

## Run locally (Linux / macOS)

Requirements: Docker (for PostgreSQL 17 + pgvector), Python 3.12+, Node 22+. Optional: Tesseract for OCR (`sudo apt install tesseract-ocr tesseract-ocr-eng` on Ubuntu, `brew install tesseract` on macOS).

```bash
scripts/setup_local.sh   # once: creates .env, starts the db on 127.0.0.1:5433, venv (CPU PyTorch), schema, npm ci
scripts/run_local.sh     # every time: db + backend (auto-reload) + Vite dev server; Ctrl+C stops
```

Open http://127.0.0.1:5173. API docs: http://127.0.0.1:8001/docs. Health check: http://127.0.0.1:8001/health. Optionally pre-download the models (~700 MB) with `PYTHONPATH=backend .venv/bin/python backend/download_models.py`.

Manual equivalent of the setup script:

```bash
cp .env.example .env
docker compose up -d db
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
.venv/bin/python -m pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/python backend/init_db.py --indexes
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
# second terminal
npm --prefix frontend ci && npm --prefix frontend run dev
```

A fresh database is empty. The original 91-document corpus lived in another machine's Docker volume. To move it, run `docker exec multimodal-retrieval-db pg_dump -U app_user -Fc multimodal_retrieval > atlas.dump` on the old machine and copy its `data/` directory. Then restore on this one with `docker exec -i multimodal-retrieval-db pg_restore -U app_user -d multimodal_retrieval --clean --if-exists < atlas.dump`. Alternatively, re-upload the PDFs through Library. `pdf_match_report.tsv` contains the old machine's macOS paths; regenerate it with `PYTHONPATH=backend .venv/bin/python scripts/match_pdfs.py` before using `restore_sources.py`.

The first use of a model downloads weights if they are not cached. Text uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions); images and image queries share `openai/clip-vit-base-patch32` (512 dimensions). Models are loaded lazily and frozen; review never fine-tunes either encoder. CLIP text queries are truncated to 77 tokens.

Scanned PDFs require Tesseract and English language data (`brew install tesseract` on macOS). If OCR is unavailable, ordinary selectable text still works. Ingestion fails clearly for PDFs with neither readable text nor extractable figures.

`backend/requirements-tested.txt` records the exact installed environment used for verification. It is an environment snapshot, including development packages; the smaller `requirements.txt` declares supported ranges. `frontend/package-lock.json` fixes frontend dependency versions.

## Run the full stack with Docker (local)

```bash
docker compose up --build
```

This uses `docker-compose.yml` plus the automatically loaded `docker-compose.override.yml`, which publishes the db (5433), the backend (8001) and the frontend (http://127.0.0.1:5173) on loopback only. Compose keeps the `postgres_data` volume, mounts `data/` for source files and models, and mounts `experiments/results/` for reports. It creates missing tables and indexes without deleting documents. Stop `scripts/run_local.sh` first, because both use ports 8001 and 5173.

## Deploy on the web (any Linux VPS)

The production stack is Caddy (automatic HTTPS) → nginx (frontend and login) → FastAPI → PostgreSQL. Only Caddy publishes ports (80/443); the database and backend are unreachable from outside. The app has no user accounts, so production **requires** an HTTP basic-auth login for the whole site. `/health` stays public for uptime checks.

Server requirements: Docker with Compose v2, at least 4 GB RAM (both models load into memory, about 1.5 GB), and enough disk for images, models and documents (~10 GB plus your corpus). For HTTPS, point a domain's DNS A record at the server and open ports 80 and 443.

```bash
git clone <your-repo-url> atlas && cd atlas
cp .env.example .env
# Edit .env:
#   POSTGRES_PASSWORD=$(openssl rand -hex 24)   and update DATABASE_URL to match (used only outside Docker)
#   SITE_ADDRESS=atlas.example.com                (or :80 for plain HTTP by IP)
#   BASIC_AUTH_USER=... BASIC_AUTH_PASSWORD=...
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend python backend/download_models.py  # optional warm-up
```

- **Updating:** `git pull`, then run the same `up -d --build` command. Data in `data/` and the `postgres_data` volume persists.
- **Logs:** run `docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f backend`.
- **Backups:** dump the database with `docker exec multimodal-retrieval-db pg_dump -U app_user -Fc multimodal_retrieval > atlas.dump`, and copy `data/` and `experiments/results/`.
- **Running scripts on the server:** use `docker compose ... exec backend python scripts/import_corpus.py ...`.
- **GPU host:** set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124` in `.env`. GPU access in containers also needs the NVIDIA container toolkit.
- **Database password:** it is fixed when the volume is first created. Choose it before the first `up`.

The backend deliberately runs a single Uvicorn worker, because experiment-job tracking and latency stats are process-local. Ingestion is synchronous, so large uploads hold a request open; the proxies allow up to 30 minutes and 500 MB. Free hosts without persistent disks or with under 2 GB RAM (e.g. Render free tier, Vercel) cannot run this stack.

## Use the application

- **Search:** choose keyword, semantic, hybrid (reciprocal-rank fusion), or CLIP figures. Results include filenames and page numbers. Conversation mode appends the last two user queries for refinement. This is a retrieval interface; it does not require a paid generative-model API or synthesize uncited answers.
- **Library:** upload PDFs, PNG/JPEG/WebP images, or paste text. Uploads retain source files. PDFs are chunked and embedded; raster figures and vector drawing clusters are extracted and CLIP-embedded. Record the original URL and actual reuse license in the upload form.
- **Legacy documents:** existing passages are preserved. Use **Attach PDF** on a library record with the original filename to retain its PDF and index its figures. A source can be attached once; existing passage IDs and evaluation judgments remain valid. Verify that the matching filename is in fact the correct version of the document.
- **Review:** choose passages or figures and entropy, margin, or random acquisition. Submit one of `relevant`, `duplicate`, `malformed`, `out_of_domain`, or `irrelevant`, plus an optional note. Correct earlier labels using Recent labels. Current labels are unique per item; every submission appends an audit event.
- **Classifier:** random cold start until 4 labels spanning 2+ classes are available for that modality. Logistic regression uses cached embeddings with balanced class weights and deterministic fitting. Queue predictions are updated after corrections. Text and figure classifiers are separate. Cold-start scores are absent, not fabricated confidence values.
- **Experiments:** displays saved metric reports and in-process search latency. Missing reports are explicitly pending. Median/p95 request latency includes inference and database work; the index benchmark deliberately measures a different scope.

The source/figure files and their metadata commit together; a failed embedding/ingestion rolls back its database transaction and removes the newly created files. Original documents/passages are never automatically deleted. `init_db.py` adds new tables and optional indexes; it is not a general migration framework for future column changes.

## Curate a reproducible corpus

Do not infer a reuse license from arXiv hosting alone. Record each source's actual license. Copy `experiments/datasets/corpus_manifest.example.json` and supply local PDF paths, source URLs, and licenses.

```bash
PYTHONPATH=backend .venv/bin/python scripts/import_corpus.py path/to/manifest.json
PYTHONPATH=backend .venv/bin/python scripts/import_corpus.py path/to/manifest.json --apply
```

The first command previews and validates; `--apply` ingests. Imports skip matching source SHA-256 hashes. Use the UI or `PATCH /api/v1/documents/{id}/provenance` to correct provenance later. The importer does not download arbitrary URLs.

Prepare controlled anomaly candidates without changing your corpus:

```bash
PYTHONPATH=backend .venv/bin/python scripts/prepare_anomalies.py --passage-id 1
```

This exports original/duplicate/corrupted/out-of-domain/irrelevant/ambiguous candidate texts with **unverified** suggested labels. Review them manually, add permitted texts through Library, and label through Review. Suggested labels are never silently imported as ground truth. Use multiple independent source documents per label. Define the intended technical domain before judging out-of-domain content. Duplicate status can require context that identical frozen embeddings alone cannot distinguish.

## Experiments

### Retrieval quality

Edit `experiments/datasets/retrieval_queries.json` using the text/image schema in `retrieval_queries.example.json`. Independently judge a fixed query set, record all relevant passage/figure IDs, and mark `judgments_verified: true` only after review. Existing judgments cover three text queries and are provisional. Text/image IDs are distinct. Use source IDs from API responses, not result ranks. Incomplete relevance judgments underestimate recall and may misjudge valid results.

```bash
PYTHONPATH=backend .venv/bin/python experiments/evaluate_retrieval.py
# Optional: --dataset path/to/heldout.json --base-url http://127.0.0.1:8001
```

Reports Precision@5, Recall@5, MRR truncated to the returned five results, and NDCG@5. Text queries run keyword, Sentence-BERT, and hybrid retrieval; image queries run CLIP. An empty image query set does not evaluate CLIP quality. Reports include failures and a provisional-judgment flag.

### Active learning

Manually label at least 12 items spanning 2+ classes and enough independent source documents to form group-disjoint train/test sets. More labels (e.g. 50–100+) and more judged queries are needed for persuasive conclusions; the minimum is only an execution prerequisite.

```bash
PYTHONPATH=backend .venv/bin/python experiments/evaluate_active_learning.py --budgets 5 10 20 40 --seeds 5
# Separate experiment for image embeddings:
PYTHONPATH=backend .venv/bin/python experiments/evaluate_active_learning.py --item-type figure
```

Uses a fixed held-out group split across all strategies and seeds. Documents sharing exact normalized passage content (or exact figure bytes) are grouped together. Uses the same one-example-per-class bootstrap per seed and equal cumulative label budgets for random, entropy, and margin selection. Acquisition sees embeddings/probabilities, then reveals the selected item's existing human label. Reports held-out macro F1, per-seed results, means/std, split IDs/groups, and an embedding fingerprint. Budgets exceeding the actual pool are omitted.

This is retrospective replay on a labeled pool; online acquisition can bias which items were labeled. Exact duplicate grouping does not detect all near duplicates. It measures whether uncertainty selection helps rather than assuming an improvement. The script refuses insufficient labels or an invalid split and writes no invented report.

### Index performance

```bash
PYTHONPATH=backend .venv/bin/python experiments/benchmark_index.py --sizes 100 500 1000 5000 --repeats 10
```

Copies actual cached text embeddings into an isolated PostgreSQL temporary table for each corpus size. Creates a cosine HNSW index there, forces exact or approximate plans, verifies plans with EXPLAIN, warms queries, and measures median/p95 query-plus-transfer latency and Recall@K against exact results. Encoding and index construction are excluded. Sizes are capped at available passages; data are never replicated to imply a larger corpus. Application data/indexes are not changed. Small samples and ties among duplicate vectors limit conclusions.

Reports live in `experiments/results/` and appear in the dashboard after Refresh results. Save a copy with your submitted report; generated result files are gitignored to keep machine-specific output out of source control.

## Verification

```bash
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests -q
npm --prefix frontend run build
docker compose config --quiet
```

Run the tests against the local database only (`DATABASE_URL` in `.env`). They create and drop temporary schemas, which is safe, but they should not be pointed at a production server.

The tests cover transaction rollback, bulk partial failure, source/figure previews, legacy attachment, image/text upload, provenance, all retrieval methods, label updates/audit events, acquisition/cold start, modality isolation, metric definitions, group-disjoint equal-budget experiments, and real model inference. PostgreSQL tests create uniquely named temporary schemas and remove only those schemas afterward. They require a running pgvector-enabled database and a role permitted to create schemas. The real-model test may download model weights on a fresh machine.

## Scope and remaining empirical work

The application's planned feature areas are implemented. The existing corpus still needs source attachment and license verification, and the project still needs human relevance judgments, anomaly labels, CLIP query judgments, and the active-learning experiment before it can support the proposal's research conclusions. See `docs/PROJECT_ASSESSMENT.md` for the before/after mapping and current evidence.

This is a local, single-workspace research prototype with synchronous ingestion and request-time logistic fitting. It has no accounts, multi-tenant isolation, distributed job queue, or production deployment hardening. OCR/diagram extraction is heuristic: inspect figure crops and label malformed cases. Search latency telemetry is bounded to the last 1,000 requests in one server process and resets on restart.

## Complete labeling and evaluations in the UI

See `docs/LABELING_GUIDE.md` for the human steps. Experiments now includes readiness checks, pooled text/image relevance judgments, explicit human verification, JSON exports, and background experiment controls. The Library now supports editing source URLs/licenses. Review supports skipping ambiguous items without creating labels.

The research APIs are under `/api/v1/research`: `/readiness`, `/queries`, `/queries/{id}/pool`, `/queries/{id}/judgments`, `/queries/{id}/verify`, `/queries/export`, `/labels/export`, `/runs`, and `/runs/{id}`. Judgment corrections invalidate verification. Retrieval runs snapshot only verified queries. Jobs use an allowlist of scripts, one running job per server process, a 30-minute timeout, and a bounded history. Tracking is process-local; do not restart the backend while an experiment is running.

Legacy source recovery can use the existing match report without scanning unrelated directories:

```bash
PYTHONPATH=backend .venv/bin/python scripts/restore_sources.py
PYTHONPATH=backend .venv/bin/python scripts/restore_sources.py --apply --limit 10
```

Recovery requires an unambiguous readable PDF with page text matching stored passage samples. It skips attached records and leaves ambiguous/mismatched sources for human selection. Existing IDs and passage embeddings remain unchanged. It does not infer reuse licenses.

## Extract text from uploaded images

PNG/JPEG/WebP uploads now run local Tesseract OCR automatically **before** resizing the visual preview. Original image resolution and EXIF orientation are preserved in the retained source. Transparent images are composited onto white for OCR. Readable text is chunked and embedded into normal passages, making it available to keyword, semantic, hybrid search, conversation refinement, and passage review. CLIP visual search remains available alongside it.

Use **Library → Image text** to view/select/copy extracted text. For PDF figure crops, open Image text from Library or **Extract image text** on a figure search result, select the crop, and click **Extract & index text**. Figure OCR is on demand; existing PDFs and the full figure corpus are not automatically reprocessed.

- `POST /api/v1/documents/upload-image` returns `ocr: {figure_id, text, status, message}` plus `passages_created`.
- `GET /api/v1/documents/{id}/ocr` returns the document's stored image OCR outcomes.
- `POST /api/v1/figures/{id}/ocr` extracts/caches text and indexes it. `?retry=true` retries previous unavailable/failed/timed-out attempts. Successful and no-text outcomes are cached; repeated clicks do not duplicate passages or invalidate labels.

The default language is English (`OCR_LANGUAGE=eng`). Install Tesseract plus the desired language data; set `TESSERACT_CMD` if the executable is outside PATH. Docker already installs English Tesseract. No extra Python dependency is needed. Requests have a 60-second OCR limit; uploaded images are limited to 20 MB and 20 megapixels. OCR failures, missing executables/language data, and no readable text are reported explicitly while keeping the image indexed for visual search. An embedding/storage failure rolls back both image and text ingestion rather than leaving partially indexed data.

OCR is heuristic and can misread small, handwritten, rotated, or low-contrast text. Human class labels are not needed to extract text. The latest verification passed 22 tests, including actual Tesseract extraction from a generated image, retrieval of the extracted text, full-resolution retention, retry/idempotency, and transaction rollback.
