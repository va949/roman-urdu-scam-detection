"""Robustness suite: score a model on perturbed copies of its test set and on hand-written probes.

- Perturbed test: every test message rewritten by one perturbation family (see perturb.py).
  SEEN families may also be used for training augmentation; UNSEEN ones never are.
- Probes: data/probes/handwritten_v1.csv, 40 short messages written by hand in styles and
  scam types the dataset lacks (synthetic, not collected from real inboxes).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

import perturb
from metrics import full

PROBES = Path(__file__).resolve().parent.parent / "data" / "probes" / "handwritten_v1.csv"
ROWS = ["clean", *perturb.SEEN, *perturb.UNSEEN, "probes"]


def evaluate(predict_proba, thr, texts, y, seed, clean_probs=None):
    """predict_proba(list[str]) -> np.ndarray of scam probabilities. Returns {row: metrics}."""
    texts, y = list(texts), np.asarray(y)
    out = {"clean": full(y, clean_probs if clean_probs is not None else predict_proba(texts), thr)}
    for name, fn in perturb.FAMILIES.items():
        rng = random.Random(f"{seed}-{name}")
        pert = [fn(t, rng) for t in texts]
        out[name] = {**full(y, predict_proba(pert), thr),
                     "changed": round(float(np.mean([a != b for a, b in zip(texts, pert)])), 3)}
    if PROBES.exists():
        pr = pd.read_csv(PROBES)
        out["probes"] = full(pr["label"].to_numpy(), predict_proba(pr["text"].tolist()), thr)
    return out


def table(per_seed: list[dict]) -> str:
    """Mean over seeds, one row per test condition."""
    lines = [f"{'condition':<14}{'macro-F1':>10}{'accuracy':>10}{'recall':>9}{'false al.':>11}{'ROC-AUC':>9}"]
    for row in ROWS:
        rs = [r[row] for r in per_seed if row in r]
        if not rs:
            continue
        m = lambda k: np.mean([r[k] for r in rs if r[k] is not None])
        tag = " *" if row in perturb.UNSEEN else ""
        lines.append(f"{row + tag:<14}{m('macro_f1'):>10.3f}{m('accuracy'):>10.3f}{m('scam_recall'):>9.3f}"
                     f"{m('false_alarm_rate'):>11.3f}{m('roc_auc'):>9.3f}")
    lines.append("(* = perturbation family never used for augmentation)")
    return "\n".join(lines)


def summarize(per_seed: list[dict]) -> dict:
    keys = ("macro_f1", "accuracy", "scam_recall", "false_alarm_rate", "roc_auc")
    return {row: {k: (round(float(np.mean([r[row][k] for r in per_seed])), 4),
                      round(float(np.std([r[row][k] for r in per_seed])), 4)) for k in keys}
            for row in ROWS if all(row in r for r in per_seed)}
