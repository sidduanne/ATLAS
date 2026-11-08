"""Shared split preparation for readiness checks and actual experiment execution."""
import hashlib
from collections import Counter

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from app.config import data_path
from app.services.learning import training_data


def grouped_split(y, groups, test_size=0.25):
    classes = set(y)
    for seed in range(42, 242):
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(y, y, groups))
        if set(y[train]) == classes and set(y[test]) == classes and all(Counter(y[train])[c] >= 2 for c in classes):
            return train, test, seed
    raise ValueError("Need more independently sourced labeled documents in each class for a group-disjoint held-out split")


def learning_dataset(db, item_type):
    rows = training_data(db, item_type)
    parents = {item.document_id: item.document_id for item, _ in rows}
    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i
    signatures = {}
    for item, _ in rows:
        content = " ".join(item.content.lower().split()).encode() if item_type == "passage" else data_path(item.storage_path).read_bytes()
        signature = hashlib.sha256(content).hexdigest()
        if signature in signatures:
            parents[root(item.document_id)] = root(signatures[signature])
        signatures[signature] = item.document_id
    return (np.array([item.embedding for item, _ in rows]), np.array([label.label for _, label in rows]),
            [item.id for item, _ in rows], np.array([root(item.document_id) for item, _ in rows]))
