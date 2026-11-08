"""Evaluate real, manually judged text and image queries through the public API."""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
TOP_K = 5


def precision_at_k(ranked_ids, relevant_ids, k=TOP_K):
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(ranked_ids, relevant_ids, k=TOP_K):
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0.0


def reciprocal_rank(ranked_ids, relevant_ids):
    return next((1/rank for rank, pid in enumerate(ranked_ids, 1) if pid in relevant_ids), 0.0)


def ndcg_at_k(ranked_ids, relevant_ids, k=TOP_K):
    seen = set()
    dcg = 0.0
    for rank, item in enumerate(ranked_ids[:k], 1):
        if item in relevant_ids and item not in seen:
            dcg += 1 / math.log2(rank+1)
        seen.add(item)
    ideal = sum(1 / math.log2(rank+1) for rank in range(1, min(len(relevant_ids), k)+1))
    return dcg / ideal if ideal else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/retrieval_queries.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text())
    queries = dataset.get("queries", [])
    if not queries:
        parser.error("Add manually judged queries to the dataset first")
    rows, failures = [], []
    for item in queries:
        modality = item.get("modality", "text")
        if modality not in {"text", "image"}:
            parser.error(f"Unsupported modality for {item['id']}")
        key = "figure_id" if modality == "image" else "passage_id"
        relevant = set(item["relevant_figure_ids" if modality == "image" else "relevant_passage_ids"])
        methods = {"clip": "/api/v1/search/images"} if modality == "image" else {
            "keyword": "/api/v1/search/keyword", "semantic": "/api/v1/search", "hybrid": "/api/v1/search/hybrid"}
        for method, endpoint in methods.items():
            try:
                with urlopen(args.base_url + endpoint + "?" + urlencode({"q": item["query"], "limit": TOP_K}), timeout=180) as response:
                    results = json.load(response)["results"]
                ranked = [r[key] for r in results]
                rows.append({"query_id": item["id"], "method": method, "query": item["query"], "modality": modality,
                             "ranked_ids": " ".join(map(str, ranked)),
                             "precision_at_5": precision_at_k(ranked, relevant), "recall_at_5": recall_at_k(ranked, relevant),
                             "mrr": reciprocal_rank(ranked, relevant), "ndcg_at_5": ndcg_at_k(ranked, relevant)})
            except Exception as exc:
                failures.append(f"{item['id']} / {method}: {exc}")
    averages = {}
    for method in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == method]
        averages[method] = {metric: sum(r[metric] for r in group)/len(group)
                            for metric in ("precision_at_5", "recall_at_5", "mrr", "ndcg_at_5")}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "top_k": TOP_K, "query_count": len(queries),
              "successful_evaluations": len(rows), "failure_count": len(failures), "failures": failures,
              "provisional_relevance_judgments": not dataset.get("judgments_verified", False),
              "image_query_count": sum(q.get("modality") == "image" for q in queries),
              "macro_averages": averages, "dataset": str(args.dataset.resolve()),
              "note": "MRR is truncated to the five retrieved results. Modalities use distinct relevant-ID spaces."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_summary.json").write_text(json.dumps(report, indent=2))
    if rows:
        with (args.output_dir / "retrieval_per_query.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
