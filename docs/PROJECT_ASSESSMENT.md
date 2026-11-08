> Follow-up update: 66 source PDFs have now been restored and 2,034 figure crops indexed. The application now includes human relevance judging, experiment readiness/execution, label exports, metadata editing, and skipping ambiguous items. Seventeen tests pass. The original assessment below describes the first implementation pass; see LABELING_GUIDE.md for the current human steps.

# Project assessment and implementation handoff

Assessed against the supplied **Multimodal Document Retrieval & Active Learning Engine** proposal and every source area in `/Users/macos/multimodal-retrieval-engine` on 30 October 2026.

The repository began as a working text-retrieval backend, rather than the complete proposed application. Across the 16 areas below, **4 were implemented, 2 were partial, and 10 were missing**. This is a feature-area count, not a percentage of engineering effort. Roughly one-third of the proposed scope had an implementation, with most UI, multimodal, review, and experimental work still absent.

The principal missing application and experiment components now have implementations. **The research project is still not empirically finished:** the current corpus needs original sources, license records, human anomaly labels, and independently judged text/image queries. Functional tests cannot substitute for those data or establish an active-learning benefit.

## Requirement-by-requirement mapping

| Area | Before this work | Current implementation and evidence |
| --- | --- | --- |
| PDF extraction, OCR fallback, chunking | Implemented | Preserved; tested with valid/malformed PDFs and rollback |
| Sentence-BERT frozen passage embeddings | Implemented | Preserved and explicitly frozen; real 384-dimensional inference tested |
| PostgreSQL + pgvector metadata/vector storage | Implemented | Preserved documents/passages; additive source, figure, review, and audit tables |
| Keyword, semantic, and hybrid text retrieval | Implemented | Preserved; integration tests and real-corpus evaluation run |
| Retrieval evaluation | Partial: 3 provisional text queries | Text and image judgment formats, P@5/R@5/MRR/NDCG, explicit failure/provisional reporting; real text evaluation rerun; image judgments pending |
| Reproducible setup | Partial: database Compose and requirements | README, environment example, frontend lockfile, tested Python environment snapshot, stack Dockerfiles, health checks, tests; full Docker image build not exercised |
| Figure extraction | Missing | Embedded raster image crops and vector drawing cluster crops, page/caption metadata, stored previews; tested |
| CLIP cross-modal search | Missing | Frozen CLIP image/text encoders, normalized 512-dimensional vectors and pgvector cosine retrieval; real inference and image-query API tested |
| React + TypeScript interface | Missing: empty frontend directory | Search, Library, Review, Experiments, upload and preview dialogs; production build and desktop/mobile browser checks |
| Original source retention/previews | Missing: temporary uploads removed | Owned source assets and checksum/provenance; browser preview; legacy PDF attachment preserves passage IDs |
| Conversational search refinement | Missing | Search turns carry the last two queries into retrieval; filenames/page citations; no invented document answers |
| Human review workflow | Missing | Separate text/figure queues, five labels, notes, label correction, append-only audit events |
| Lightweight classifier and acquisition | Missing | Logistic regression on cached embeddings; entropy, margin, random; cold-start behavior and correction updates tested |
| Review and latency analytics | Missing | Corpus/label/action counts, bounded process-local median/p95 search latency, saved experiment reports |
| Active-learning comparison | Missing | Equal-budget replay with fixed document/duplicate-disjoint holdout, seeds, macro F1 curves; algorithm tests pass; real run correctly refuses absent labels |
| Exact versus HNSW experiment | Missing | Isolated temporary-table benchmark at increasing real corpus sizes, warmup, EXPLAIN verification, median/p95 and Recall@K; run at 100/500/1,000 passages |

## Existing data and preservation

The database contains **91 documents and 6,409 passages**. After implementation and testing, those counts remain unchanged. It has **0 retained source files, 0 figures, and 0 human labels**. The new tests create separate, uniquely named PostgreSQL schemas; their sample documents, figures, and labels are removed with those test schemas. No generated test labels were added to the user's corpus.

The new initializer adds tables and indexes without deleting/rebuilding existing records. The Library's Attach PDF action enables source retention and CLIP figure indexing for earlier records. It checks the filename and refuses replacement of an existing source; a matching filename still requires verifying the correct document version.

At the start, the Git repository had no commits and the supplied application files were untracked. No commit was created or existing user file deleted. New code is saved directly in the supplied repository. The existing `.env`, document/passages data, matching report, and source backup file were preserved.

## Verification performed

- **12 automated tests passed**, including an end-to-end PDF ingestion/search flow with the actual frozen Sentence-BERT and CLIP models. Fast tests substitute deterministic vectors; the explicit real-model test does not.
- **React/TypeScript production build passed** (`tsc -b && vite build`).
- **Docker Compose configuration validated**. The complete images were not built/run during this work.
- **Python compilation check passed** for application, scripts, and experiments.
- **Live backend health check passed** against the existing PostgreSQL database.
- Browser checks covered responsive rendering, real hybrid search results, source-preview recovery for legacy documents, library filtering, review cold start, and dashboard reports. UI labeling was not performed on real items without human judgments; its write paths are covered in the isolated API tests.
- Third-party PyMuPDF/SWIG deprecation warnings remain in the test output; the test run has no failing checks.

### Fresh retrieval results

The existing three-query dataset remains provisional and includes no image queries. All 9 method/query requests completed without failure. Each text method currently yields macro P@5 **0.467**, R@5 **0.583**, and MRR@5 **0.667**. NDCG differs slightly by method. These are measurements against a very small, incompletely judged set, not validated retrieval-quality conclusions. The older saved scores were higher; the current larger corpus and provisional judgments do not support an improvement claim.

### Initial index benchmark

Three text queries, three repeats, nine measured queries per size/mode, after warmup:

| Corpus passages | Exact median / p95 (ms) | HNSW median / p95 (ms) | HNSW Recall@5 |
| --- | --- | --- | --- |
| 100 | 0.381 / 0.949 | 0.853 / 1.476 | 1.000 |
| 500 | 0.464 / 1.122 | 0.467 / 1.634 | 1.000 |
| 1,000 | 0.926 / 1.573 | 0.768 / 1.752 | 1.000 |

These are small local warm-query measurements. They exclude embedding inference and index construction. HNSW is not consistently faster at these small sizes, and nine samples are insufficient for strong p95 conclusions. EXPLAIN confirmed the intended search plan. The benchmark uses temporary tables and leaves application data and indexes intact.

### Active-learning experiment

The real-corpus command was attempted. It correctly exited with: “Review at least 12 items across 2+ classes and multiple source documents first.” It did not generate a result file. Equal-budget simulation, fixed holdout behavior, and reproducibility were verified using explicitly synthetic test data; those test scores are not research results.

## Work still required for the proposal deliverable

1. **Restore the corpus's sources.** Attach each original PDF from Library or ingest a curated manifest. Verify source URLs and reuse licenses; they cannot be inferred from document filenames.
2. **Inspect figure extraction.** Check page/caption alignment and raster/vector crops. Review incomplete or malformed figures. Heuristic drawing clusters do not guarantee perfect scientific figure segmentation.
3. **Create independent retrieval judgments.** Expand and freeze a held-out query set, record relevant passage and figure IDs, and verify the provisional judgments. Rerun text and CLIP quality evaluation.
4. **Collect human anomaly labels.** Use multiple documents per class, include controlled duplicates, irrelevant/out-of-domain, corrupted, and ambiguous examples. A helper prepares unverified candidates without inserting labels.
5. **Run the active-learning experiment.** Start with the 12-item minimum only to exercise the pipeline; collect substantially more labels and independent documents for credible learning curves. Compare strategies under the same budgets and report whether there is a benefit.
6. **Strengthen index evidence.** Run more repeats/queries and larger actual corpus sizes, record hardware and library versions, and report the measured speed/recall trade-off.
7. **Write the project results/discussion.** Interpret the held-out metrics, extraction errors, duplicate ambiguity, query/judgment limitations, and retrospective labeling bias. No claim of improved F1 or label efficiency is established yet.

## Main implementation locations

- `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/src/style.css`: React workspace and responsive styling.
- `backend/app/api/workspace.py`: ingestion additions, source/figure APIs, chat/refinement, review, analytics, reports.
- `backend/app/models/review.py`: additive source/figure/current-label/audit tables.
- `backend/app/services/assets.py`, `clip.py`, `learning.py`: source files, extraction, frozen CLIP and classifier/acquisition.
- `backend/app/main.py`, `services/ingestion.py`, `services/embeddings.py`, `backend/init_db.py`: integration with the existing backend.
- `experiments/evaluate_retrieval.py`, `evaluate_active_learning.py`, `benchmark_index.py`: executable experiment runners.
- `scripts/import_corpus.py`, `prepare_anomalies.py`: curation and controlled-candidate preparation.
- `backend/tests/`: isolated PostgreSQL integration tests and experiment/model checks.
- `README.md`: setup, operation, experiment commands, methodology, and limitations.

## Run now

The existing backend is serving at `http://127.0.0.1:8001` and the new frontend is running at **http://127.0.0.1:5173**. Start/restart instructions are in README. This is a local single-workspace prototype with synchronous ingestion and request-time classifier fitting. Accounts, multi-tenant isolation, production job orchestration, and public deployment were outside the proposal's prototype scope and have not been added.
