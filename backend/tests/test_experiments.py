import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from evaluate_retrieval import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k
from evaluate_active_learning import grouped_split, simulate


def test_retrieval_metrics_and_duplicates():
    assert precision_at_k([1, 1, 2], {1, 2}, 5) == 0.4
    assert recall_at_k([1, 1], {1, 2}, 5) == 0.5
    assert recall_at_k([], set()) == 0
    assert reciprocal_rank([3, 2, 1], {1}) == 1/3
    assert ndcg_at_k([1, 2], {1, 2}) == 1
    assert ndcg_at_k([1, 1], {1, 2}) < 1


def test_acquisition_equal_budgets_fixed_holdout_and_reproducibility():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 12))
    y = np.array(['relevant', 'irrelevant'] * 40)
    groups = np.repeat(np.arange(20), 4)
    train, test, seed = grouped_split(y, groups)
    assert not set(groups[train]) & set(groups[test])
    assert not set(train) & set(test)
    curves, records = simulate(x, y, train, test, [2, 5, 10], 2)
    assert len(curves) == 9 and len(records) == 18
    initial = [r['f1'] for r in records if r['labels_reviewed'] == 2 and r['seed'] == 0]
    assert len(set(initial)) == 1
    assert simulate(x, y, train, test, [2, 5, 10], 2) == (curves, records)
    assert all(0 <= r['mean_f1'] <= 1 for r in curves)


def test_insufficient_independent_groups_rejected():
    with pytest.raises(ValueError):
        grouped_split(np.array(['a', 'b']*10), np.zeros(20))
