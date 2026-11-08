# ATLAS: Multimodal Document Retrieval & Active Learning Engine

A research prototype that searches technical PDFs, images and text, and learns from human review.

- **Search:** four methods. **Keyword** is PostgreSQL full-text search. **Semantic** uses Sentence-BERT embeddings. **Hybrid** combines the two with reciprocal-rank fusion. **Figure search** uses CLIP to match a text description against extracted images. A conversation mode refines a search using your previous messages.
- **Review:** people label passages and figures. A lightweight classifier, trained on frozen embeddings, then decides which uncertain item to show next. This is active learning.
- **Experiments:** measure retrieval quality, label efficiency (uncertainty sampling against random selection), and the speed and recall of an exact index against an HNSW index.

Stack: React + TypeScript, FastAPI, PostgreSQL + pgvector, MiniLM (384-d text vectors), CLIP (512-d image/text vectors) and Tesseract OCR, all in Docker.

## Quick start

You need only **Git** and **Docker**: Docker Engine with the Compose plugin on Linux, Docker Desktop on macOS or Windows. Give Docker at least **4 GB RAM** (Docker Desktop → Settings → Resources) and about **12 GB of disk**.

```bash
git clone <your-repo-url> atlas
cd atlas
cp .env.example .env          # Windows PowerShell: copy .env.example .env
docker compose up -d --build
```

Open **http://127.0.0.1:5173**. API docs are at `/docs` and the health check at `/health`. The first build takes 5–15 minutes, because it downloads PyTorch (CPU) and bakes both models into the image. Later starts take seconds.

| Task | Command |
| --- | --- |
| Stop (data is kept) | `docker compose down` |
| Start again | `docker compose up -d` |
| Update after `git pull` | `docker compose up -d --build` |
| Logs / status | `docker compose logs -f backend` / `docker compose ps` |
| Database shell | `docker compose exec db psql -U app_user -d multimodal_retrieval` |
| Run tests | `docker compose exec backend python -m pytest backend/tests -q` |
| Delete **everything**, including data | `docker compose down -v` |

If port 5173 is already in use, set `ATLAS_PORT` in `.env`. Only the web app is published, and only on 127.0.0.1. The database and backend are reachable only inside Docker.

**Where data lives:** Docker volumes. `postgres_data` holds documents, passages, embeddings, labels and judgments. `atlas_data` holds the original source files and figure crops. `atlas_results` holds experiment reports. The host folder `data/pdfs/` is mounted read-only for bulk imports.

**Backup and restore** (for example, to move to another machine or a server):

```bash
docker compose exec -T db pg_dump -U app_user -Fc multimodal_retrieval > atlas.dump
docker compose cp backend:/app/data ./atlas-data-backup
docker compose cp backend:/app/experiments/results ./atlas-results-backup
# On the new machine, after `docker compose up -d`:
docker compose exec -T db pg_restore -U app_user -d multimodal_retrieval --clean --if-exists < atlas.dump
docker compose cp ./atlas-data-backup/. backend:/app/data/
```

A fresh install starts with an empty library. To bring over the original 91-document corpus, use backup and restore from the machine that holds it, or upload the PDFs again through Library.

## Project layout

```
backend/app/
  main.py              app setup, error handling, search latency timing
  core.py              settings, database session, UserError, data-directory guard
  models.py            database tables + init_db()
  api/                 HTTP routes only: documents, search, review, research
  services/
    ingestion.py       PDF/text/image ingest, chunking, figure extraction, OCR
    search.py          keyword, semantic, hybrid (RRF) and CLIP retrieval
    learning.py        classifier, acquisition scores, leakage-safe splits
    encoders.py        frozen Sentence-BERT and CLIP models
backend/tests/         API tests (isolated schema, fake encoders) and experiment tests
frontend/src/          App.tsx, one file per page, dialogs.tsx, ui.tsx, api.ts, style.css
experiments/           evaluate_retrieval.py, evaluate_active_learning.py, benchmark_index.py
scripts/               import_corpus.py, prepare_anomalies.py
```

## Using the application

**Search.** Pick a method and ask in your own words. Results cite the filename and page; **View source** opens the original file at that page. With **Conversation on**, each search also includes your previous two messages. Answers are never generated: every result is a retrieved source.

**Library.**
- **Uploads.** Upload PDFs, PNG/JPEG/WebP images, or pasted text, and record the original URL and the *actual* reuse license. A hosting site alone, such as arXiv, does not establish a reuse license.
- **What gets indexed.** PDFs are split into text passages and embedded. Embedded images and clusters of vector drawings are cropped out as figures and CLIP-embedded. Scanned pages go through OCR.
- **Image text.** Uploaded images are OCR'd automatically at full resolution, and the text becomes searchable passages. For PDF figure crops, use **Image text** (or **Extract image text** on a search result).
- **Older records without a source.** **Attach PDF** adds the original file to such a record. It keeps the existing passage IDs and indexes the figures.
- **Failed uploads.** Any failed upload is rolled back completely, including files already written.

**Review (labeling).** Choose text passages or figures, and an acquisition strategy (highest entropy, smallest margin, or random). Then label each item:

| Label | Meaning |
| --- | --- |
| relevant | Readable, useful technical content within the intended domain |
| duplicate | A redundant copy of an item already in the corpus (check context/source) |
| malformed | Broken extraction, illegible text, or an unusable figure crop |
| out_of_domain | Content outside the project's declared technical domain |
| irrelevant | In the broad domain but not useful (navigation, boilerplate, …) |

- **Notes and corrections.** Add a note explaining your reasoning. If an item is ambiguous, use **Skip** rather than guessing. Correct mistakes under **Recent labels**; every submission is kept in an audit trail.
- **Cold start.** The classifier starts once there are 4 labels across 2 classes for that item type. Until then, items are shown in random order.
- **Honest labels.** Never assign wrong labels to balance the classes. If almost everything is relevant, create controlled anomaly examples with `scripts/prepare_anomalies.py` (below) and judge those.

**Experiments.** The page shows search latency, the saved reports, and a research panel:
1. **Readiness:** missing sources and licenses, label counts, and how many queries have been verified.
2. **Human relevance judgments:**
   - Write a held-out query, then use **Pool search results** to collect the top 5 from every method.
   - Mark each result **Relevant** or **Not relevant**, then **Verify** the query.
   - Changing a judgment later removes the verification.
   - These judgments are separate from the Review labels.
3. **Run experiments:** experiments run in the background, one at a time. Keep the backend running until a run finishes, then click **Refresh results**.

## Experiments (command line)

All commands run inside the backend container. Reports are written to the `atlas_results` volume and appear on the Experiments page.

**Retrieval quality:** Precision@5, Recall@5, MRR and NDCG@5. Text queries run keyword, semantic and hybrid search; image queries run CLIP.
```bash
docker compose exec backend python experiments/evaluate_retrieval.py
```
- **Default dataset.** `experiments/datasets/retrieval_queries.json` holds three provisional text queries. Each entry looks like `{"id", "query", "modality": "text"|"image", "relevant_passage_ids" | "relevant_figure_ids"}`.
- **Using your own judgments.** Pass `--dataset`, or use **Evaluate judged queries** in the UI, which takes a snapshot of your verified queries.
- **Recall is limited by the pool.** Recall only counts the judged pool, so relevant items outside it are missed.

**Active learning:** a retrospective replay over your human labels.
```bash
docker compose exec backend python experiments/evaluate_active_learning.py --budgets 5 10 20 40 --seeds 5
docker compose exec backend python experiments/evaluate_active_learning.py --item-type figure
```
- **Setup.** Every strategy and seed uses the same held-out set. Documents that share exact duplicate content are kept on the same side of the split. Each run starts from the same one-example-per-class bootstrap and uses equal label budgets.
- **Output.** The script reports macro F1 on the held-out set for each strategy and budget.
- **Minimum labels.** It needs at least **12 labels** across 2+ classes from several independent documents. That is only the minimum to run; aim for 50–100+.
- **No invented results.** The script refuses to run on insufficient data, and it never assumes uncertainty sampling helps.

**Index performance:** exact search against HNSW.
```bash
docker compose exec backend python experiments/benchmark_index.py --sizes 100 500 1000 5000 --repeats 10
```
- **Isolated.** Real passage embeddings are copied into a temporary table; your application data and indexes are untouched.
- **Checked plans.** EXPLAIN confirms that each query uses the intended plan.
- **What it measures.** Median and p95 latency, and HNSW Recall@K compared with exact search. Query encoding and index building are not timed. Sizes are capped at the number of real passages.

**Bulk import with licenses:** put PDFs and a `manifest.json` in `data/pdfs/`, for example `{"documents": [{"path": "paper.pdf", "source_url": "https://…", "license": "CC BY 4.0"}]}`. Then run:
```bash
docker compose exec backend python scripts/import_corpus.py /app/import/manifest.json          # preview
docker compose exec backend python scripts/import_corpus.py /app/import/manifest.json --apply  # ingest
```

**Anomaly candidates for labeling:**
```bash
docker compose exec backend python scripts/prepare_anomalies.py --passage-id 1
```
This writes duplicate, corrupted, out-of-domain, irrelevant and ambiguous variants of a passage, with suggested labels marked **unverified**. Judge them yourself, add the ones you keep through Library, and label them in Review.

## Deploy on the web (any Linux VPS)

The production overlay `docker-compose.prod.yml` adds Caddy for automatic HTTPS. Requests go Caddy → nginx (frontend + login) → FastAPI → PostgreSQL. The app has no user accounts, so production **requires** a site-wide login; `/health` stays public for uptime checks.

Server requirements: Docker with Compose v2, at least 4 GB RAM, and about 12 GB of disk plus your corpus. For HTTPS, point a domain's DNS A record at the server and open ports 80 and 443.

```bash
git clone <your-repo-url> atlas && cd atlas
cp .env.example .env
# Edit .env:
#   POSTGRES_PASSWORD=<output of: openssl rand -hex 24>   (choose before the first start)
#   SITE_ADDRESS=atlas.example.com                         (or :80 for plain HTTP by IP)
#   BASIC_AUTH_USER=...  BASIC_AUTH_PASSWORD=...
export COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml   # then plain `docker compose ...` works
docker compose up -d --build
```

- **Missing secrets.** The overlay refuses to start without a database password and a login.
- **One worker.** The backend runs a single Uvicorn worker on purpose, because experiment jobs and latency stats are tracked in memory.
- **Uploads and hosts.** Uploads are processed synchronously; the proxies allow up to 30 minutes and 500 MB per request. Free hosts with no persistent disk or under 2 GB RAM (e.g. Render free tier, Vercel) cannot run this stack.
- **GPU.** For an NVIDIA GPU, set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124` and install the NVIDIA container toolkit.

## Status and remaining research work

All the features in the proposal are implemented and tested: 22 automated tests, including real model inference and real Tesseract OCR. **The research results are not finished yet.** The remaining work is human work:

1. **Sources.** Attach or import the original PDFs and record their actual reuse licenses. Check the figure crops, which are found by heuristics, and label broken ones `malformed`.
2. **Labels.** Label 50–100+ passages in Review, drawn from several independent documents per class. Include controlled duplicate, irrelevant, out-of-domain and corrupted examples.
3. **Relevance judgments.** Write and verify a fixed set of held-out text and image queries. The three existing text queries are provisional, and there are no image queries yet.
4. **Experiments and write-up.** Run all three experiments and report what the numbers show. That includes the case where uncertainty sampling does *not* beat random. Also cover the limitations:
   - retrospective labeling bias;
   - near-duplicates that exact-match grouping misses;
   - recall that only counts the judged pool;
   - small benchmark samples.

Known limits of the prototype:
- It is a single-workspace system with no accounts.
- Ingestion is synchronous, and the classifier is refitted on every request.
- Experiment tracking and latency statistics reset when the backend restarts.
- Figure segmentation and OCR are heuristic.
