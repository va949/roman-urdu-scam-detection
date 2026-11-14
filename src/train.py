"""Train our own (non-LLM) model with a template-aware train/val/test split, then save it.

- Tunes the decision threshold and regularisation on VALIDATION.
- Reports TEST once at the end. Never tune on test.

Usage:
    python src/train.py --data data/raw/phishing_dataset_1000.xlsx --out models/tfidf_lr.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, make_pipeline

import data as data_mod
import splits


def build(C):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


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
    return {"roc_auc": round(float(roc_auc_score(y, prob)), 4),
            "pr_auc": round(float(average_precision_score(y, prob)), 4)}


def report(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
            "scam_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "false_alarm_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col"); ap.add_argument("--label-col"); ap.add_argument("--group-col")
    ap.add_argument("--out", default="models/tfidf_lr.joblib")
    ap.add_argument("--seed", type=int, default=0, help="split seed (same seed = same split as xlmr.py)")
    a = ap.parse_args()

    df = data_mod.load(a.data, a.text_col, a.label_col, a.group_col)
    tr, va, te = splits.train_val_test(df, seed=a.seed)
    X, y = df["text"], df["label"].to_numpy()
    print(f"train={len(tr)} val={len(va)} test={len(te)} | templates "
          f"{df.group.iloc[tr].nunique()}/{df.group.iloc[va].nunique()}/{df.group.iloc[te].nunique()}")

    # 1) pick C on validation (ties keep the smallest C, i.e. the strongest regularisation)
    best = None
    for C in (0.3, 1.0, 3.0, 10.0):
        m = build(C).fit(X.iloc[tr], y[tr])
        f1 = f1_score(y[va], m.predict(X.iloc[va]), average="macro", zero_division=0)
        print(f"  C={C:<5} val macro-F1={f1:.3f}")
        if best is None or f1 > best[1]:
            best = (C, f1, m)
    C, _, model = best

    # 2) pick the decision threshold on validation (trade recall vs false alarms)
    pv = model.predict_proba(X.iloc[va])[:, 1]
    thr = pick_threshold(y[va], pv)

    # 3) one look at the locked test set
    prob_te = model.predict_proba(X.iloc[te])[:, 1]
    result = {"C": C, "threshold": round(float(thr), 3),
              "val": report(y[va], (pv >= thr).astype(int)),
              "test": {**report(y[te], (prob_te >= thr).astype(int)), **ranking(y[te], prob_te)}}
    print(json.dumps(result, indent=2))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": float(thr), "meta": result}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
