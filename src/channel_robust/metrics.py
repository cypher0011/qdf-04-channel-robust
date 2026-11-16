"""Evaluation metrics. Convention: label 1 = fake, higher score = more fake."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold where it happens."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def summary(labels, scores) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    e, t = eer(labels, scores)
    return {"eer": e, "eer_threshold": t, "auc": float(roc_auc_score(labels, scores)),
            "n": int(len(labels)), "n_fake": int(labels.sum()), "n_real": int((1 - labels).sum())}
