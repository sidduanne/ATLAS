"""Train only the lightweight classifier; Sentence-BERT and CLIP stay frozen."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy import select

from app.models import Figure, Passage, ReviewLabel

MODELS = {"passage": Passage, "figure": Figure}


def uncertainty(probabilities, strategy="entropy"):
    p = np.asarray(probabilities, dtype=float)
    if strategy == "margin":
        ranked = np.sort(p, axis=1)
        return 1 - (ranked[:, -1] - ranked[:, -2])
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1) / np.log(p.shape[1])


def training_data(db, item_type):
    model = MODELS[item_type]
    rows = db.execute(select(model, ReviewLabel).join(
        ReviewLabel, (ReviewLabel.item_id == model.id) & (ReviewLabel.item_type == item_type)
    ).where(model.embedding.is_not(None))).all()
    return rows


def fit_classifier(rows):
    if len(rows) < 4 or len({label.label for _, label in rows}) < 2:
        return None
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(
        np.asarray([item.embedding for item, _ in rows]), [label.label for _, label in rows])


def rank_candidates(items, classifier, strategy, seed):
    if not items:
        return []
    if classifier is None or strategy == "random":
        order = np.random.default_rng(seed).permutation(len(items))
        return [(items[i], None, None, {}) for i in order]
    probabilities = classifier.predict_proba(np.asarray([item.embedding for item in items]))
    scores = uncertainty(probabilities, strategy)
    order = np.argsort(-scores, kind="stable")
    return [(items[i], float(scores[i]), str(classifier.classes_[probabilities[i].argmax()]),
             dict(zip(classifier.classes_.tolist(), probabilities[i].tolist(), strict=True))) for i in order]
