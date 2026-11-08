"""Active learning: the lightweight classifier, acquisition scores, and leakage-safe splits.

Only the logistic-regression classifier is trained; Sentence-BERT and CLIP stay frozen.
"""
import hashlib
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sqlalchemy import select

from app.core import data_path
from app.models import Figure, Passage, ReviewLabel

MODELS = {"passage": Passage, "figure": Figure}


def uncertainty(probabilities, strategy="entropy"):
    """Normalized entropy, or 1 - (top-1 minus top-2 probability) for margin."""
    p = np.asarray(probabilities, dtype=float)
    if strategy == "margin":
        ranked = np.sort(p, axis=1)
        return 1 - (ranked[:, -1] - ranked[:, -2])
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1) / np.log(p.shape[1])


def training_data(db, item_type):
    """(item, label) rows for every labeled item of this type that has an embedding."""
    model = MODELS[item_type]
    return db.execute(select(model, ReviewLabel).join(
        ReviewLabel, (ReviewLabel.item_id == model.id) & (ReviewLabel.item_type == item_type)
    ).where(model.embedding.is_not(None))).all()


def fit_classifier(rows):
    """None during cold start (fewer than 4 labels or a single class)."""
    if len(rows) < 4 or len({label.label for _, label in rows}) < 2:
        return None
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(
        np.asarray([item.embedding for item, _ in rows]), [label.label for _, label in rows])


def rank_candidates(items, classifier, strategy, seed):
    """Order unlabeled items by acquisition score: (item, score, prediction, probabilities)."""
    if not items:
        return []
    if classifier is None or strategy == "random":
        return [(items[i], None, None, {}) for i in np.random.default_rng(seed).permutation(len(items))]
    probabilities = classifier.predict_proba(np.asarray([item.embedding for item in items]))
    scores = uncertainty(probabilities, strategy)
    classes = classifier.classes_.tolist()
    return [(items[i], float(scores[i]), str(classes[probabilities[i].argmax()]),
             dict(zip(classes, probabilities[i].tolist(), strict=True))) for i in np.argsort(-scores, kind="stable")]


def grouped_split(y, groups, test_size=0.25):
    """Train/test split with no group on both sides and every class present in both."""
    classes = set(y)
    for seed in range(42, 242):
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(y, y, groups))
        if set(y[train]) == classes and set(y[test]) == classes and all(Counter(y[train])[c] >= 2 for c in classes):
            return train, test, seed
    raise ValueError("Need more independently sourced labeled documents in each class for a group-disjoint held-out split")


def learning_dataset(db, item_type):
    """Embeddings, labels, item ids and leakage groups for the labeled items.

    Items from the same document share a group; documents containing an exact duplicate
    (normalized text or identical figure bytes) are merged into one group (union-find).
    """
    rows = training_data(db, item_type)
    parents = {item.document_id: item.document_id for item, _ in rows}

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    signatures = {}
    for item, _ in rows:
        content = (" ".join(item.content.lower().split()).encode() if item_type == "passage"
                   else data_path(item.storage_path).read_bytes())
        signature = hashlib.sha256(content).hexdigest()
        if signature in signatures:
            parents[root(item.document_id)] = root(signatures[signature])
        signatures[signature] = item.document_id
    return (np.array([item.embedding for item, _ in rows]), np.array([label.label for _, label in rows]),
            [item.id for item, _ in rows], np.array([root(item.document_id) for item, _ in rows]))
