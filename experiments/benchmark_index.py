"""Benchmark actual pgvector exact/HNSW plans in an isolated temporary table.

No changes to application indexes or corpus. Query encoding is excluded from timing.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from sqlalchemy import select, text

from app.core import SessionLocal, engine
from app.models import Passage
from app.services.encoders import embed_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000, 5000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "datasets/retrieval_queries.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/index_benchmark.json")
    args = parser.parse_args()
    if args.k < 1 or args.repeats < 1 or not 1 <= args.ef_search <= 1000 or not args.sizes or min(args.sizes) < args.k:
        parser.error("Use positive repeats, 1 <= ef-search <= 1000, and sizes >= k >= 1")
    dataset = json.loads(args.dataset.read_text())
    queries = [q["query"] for q in dataset["queries"] if q.get("modality", "text") == "text"]
    if not queries:
        parser.error("Need at least one text query")
    vectors = [embed_query(q) for q in queries]
    with SessionLocal() as db:
        rows = db.execute(select(Passage.id, Passage.embedding).where(Passage.embedding.is_not(None)).order_by(Passage.id)).all()
    if len(rows) < args.k:
        parser.error("Not enough embedded passages")
    sizes = sorted({min(size, len(rows)) for size in args.sizes})
    measurements = []
    for size in sizes:
        with engine.begin() as conn:
            conn.execute(text("CREATE TEMP TABLE atlas_index_benchmark (id integer PRIMARY KEY, embedding vector(384)) ON COMMIT DROP"))
            conn.execute(text("INSERT INTO atlas_index_benchmark (id, embedding) VALUES (:id, CAST(:embedding AS vector))"),
                         [{"id": pid, "embedding": "[" + ",".join(map(str, vector)) + "]"} for pid, vector in rows[:size]])
            conn.execute(text("CREATE INDEX atlas_benchmark_hnsw ON atlas_index_benchmark USING hnsw (embedding vector_cosine_ops)"))
            conn.execute(text("ANALYZE atlas_index_benchmark"))
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {args.ef_search}"))
            sql = "SELECT id FROM atlas_index_benchmark ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
            params = [{"q": "[" + ",".join(map(str, v)) + "]", "k": args.k} for v in vectors]
            gold = []
            for mode in ("exact", "hnsw"):
                conn.execute(text(f"SET LOCAL enable_indexscan = {'off' if mode == 'exact' else 'on'}"))
                conn.execute(text("SET LOCAL enable_bitmapscan = off"))
                conn.execute(text(f"SET LOCAL enable_seqscan = {'on' if mode == 'exact' else 'off'}"))
                plan = conn.execute(text("EXPLAIN (FORMAT JSON) " + sql), params[0]).scalar()
                plan_text = json.dumps(plan)
                if mode == "hnsw" and "atlas_benchmark_hnsw" not in plan_text:
                    raise RuntimeError("Planner did not use HNSW; benchmark would be invalid")
                if mode == "exact" and "atlas_benchmark_hnsw" in plan_text:
                    raise RuntimeError("Exact baseline unexpectedly used HNSW")
                for q in params:  # warmup, excluded from measurements
                    conn.execute(text(sql), q).all()
                timings, recalls = [], []
                for repeat in range(args.repeats):
                    for i, q in enumerate(params):
                        started = perf_counter()
                        result = conn.execute(text(sql), q).scalars().all()
                        timings.append((perf_counter()-started)*1000)
                        if mode == "exact" and repeat == 0:
                            gold.append(set(result))
                        recalls.append(len(set(result) & gold[i]) / len(gold[i]))
                measurements.append({"corpus_size": size, "mode": mode, "median_ms": float(np.median(timings)),
                                     "p95_ms": float(np.percentile(timings, 95)), "recall_at_k": float(np.mean(recalls)),
                                     "samples": len(timings), "query_plan": plan})
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "k": args.k, "query_count": len(queries),
              "available_passages": len(rows), "repeats": args.repeats, "ef_search": args.ef_search,
              "measurements": measurements, "timing_scope": "warm database query and result transfer; excludes embedding inference and index build",
              "corpus_selection": "ascending passage IDs, capped at available real corpus; no synthetic size inflation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "sizes": sizes, "measurements": [{k:v for k,v in r.items() if k != 'query_plan'} for r in measurements]}, indent=2))


if __name__ == "__main__":
    main()
