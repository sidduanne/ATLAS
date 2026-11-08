"""Replay acquisition against real human labels on frozen cached embeddings.

The held-out split is fixed across strategies and seeds. Source-document groups and
exact content duplicates are joined before splitting to prevent common leakage.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from app.core import SessionLocal
from app.services.learning import grouped_split, learning_dataset, uncertainty


def simulate(x, y, train, test, budgets, seeds):
    records = []
    classes = sorted(set(y))
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        # Identical, disclosed class-balanced bootstrap for every strategy.
        initial = [int(rng.choice(train[y[train] == label])) for label in classes]
        for strategy in ("random", "entropy", "margin"):
            selected = list(initial)
            remaining = [int(i) for i in train if i not in selected]
            acquisition_rng = np.random.default_rng(seed)
            while True:
                model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(x[selected], y[selected])
                if len(selected) in budgets:
                    score = f1_score(y[test], model.predict(x[test]), labels=classes, average="macro", zero_division=0)
                    records.append({"seed": seed, "strategy": strategy, "labels_reviewed": len(selected), "f1": float(score)})
                if len(selected) >= max(budgets) or not remaining:
                    break
                if strategy == "random":
                    index = int(acquisition_rng.integers(len(remaining)))
                else:
                    index = int(np.argmax(uncertainty(model.predict_proba(x[remaining]), strategy)))
                selected.append(remaining.pop(index))
    curves = []
    for strategy in ("random", "entropy", "margin"):
        for budget in budgets:
            scores = [r["f1"] for r in records if r["strategy"] == strategy and r["labels_reviewed"] == budget]
            curves.append({"strategy": strategy, "labels_reviewed": budget, "mean_f1": float(np.mean(scores)), "std_f1": float(np.std(scores))})
    return curves, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-type", choices=["passage", "figure"], default="passage")
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/active_learning.json")
    args = parser.parse_args()
    with SessionLocal() as db:
        x, y, ids, groups = learning_dataset(db, args.item_type)
        if len(y) < 12 or len(set(y)) < 2:
            parser.error("Review at least 12 items across 2+ classes and multiple source documents first")
    try:
        train, test, split_seed = grouped_split(y, groups)
    except ValueError as exc:
        parser.error(str(exc))
    budgets = sorted({b for b in args.budgets if len(set(y)) <= b <= len(train)})
    if not budgets or args.seeds < 1:
        parser.error(f"Use positive seeds and budgets between {len(set(y))} and {len(train)}")
    curves, records = simulate(x, y, train, test, budgets, args.seeds)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "item_type": args.item_type,
              "labeled_count": len(y), "heldout_count": len(test), "pool_count": len(train), "seeds": args.seeds,
              "split_seed": split_seed, "bootstrap": "one randomly chosen labeled item per class, identical across strategies",
              "heldout_item_ids": [ids[i] for i in test], "pool_item_ids": [ids[i] for i in train],
              "heldout_groups": sorted(set(groups[test].tolist())), "pool_groups": sorted(set(groups[train].tolist())),
              "label_counts": dict(Counter(y)), "curves": curves, "per_seed": records,
              "embedding_fingerprint": hashlib.sha256(x.tobytes()).hexdigest(),
              "caveat": "Retrospective replay on manually labeled items; online acquisition may bias this pool. No benefit is assumed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "heldout": len(test), "curves": curves}, indent=2))


if __name__ == "__main__":
    main()
