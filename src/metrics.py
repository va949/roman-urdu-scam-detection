"""Shared metrics and the validation threshold rule (used by train.py, xlmr.py, robustness.py)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

THRESHOLDS = np.linspace(0.2, 0.8, 25)


def pick_threshold(y, p, grid=THRESHOLDS):
    """Threshold with the best validation macro-F1; ties go to the MIDDLE of the tied range.

    Validation is often saturated (every threshold scores 1.0). Taking the first
    tie would always pick 0.2, the most trigger-happy setting; the middle keeps
    the widest margin on both sides.
    """
    scores = np.array([f1_score(y, (p >= t).astype(int), average="macro", zero_division=0) for t in grid])
    return float(np.median(grid[np.isclose(scores, scores.max())]))


def ranking(y, prob):
    """Threshold-free: how well scam probabilities rank scams above legit messages."""
    if len(set(y)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": round(float(roc_auc_score(y, prob)), 4),
            "pr_auc": round(float(average_precision_score(y, prob)), 4)}


def report(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
            "accuracy": round(float((tp + tn) / (tp + tn + fp + fn)), 4),
            "scam_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def full(y, prob, thr):
    """report() at the threshold + ranking(), for a vector of scam probabilities."""
    y, prob = np.asarray(y), np.asarray(prob)
    return {**report(y, (prob >= thr).astype(int)), **ranking(y, prob)}
