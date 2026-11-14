"""TF-IDF baselines evaluated with random vs template-aware splits.

Usage:
    python src/baseline.py --data data/raw/phishing_dataset_1000.xlsx
    python src/baseline.py --data data/raw/phishing_dataset_1000.xlsx --show-columns
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

import data as data_mod
import splits


def models():
    def feats():
        # Character n-grams cope with Roman Urdu's inconsistent spelling (hai/hy/he).
        return FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ])
    return {
        "tfidf_logreg": lambda: make_pipeline(feats(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "tfidf_linearsvc": lambda: make_pipeline(feats(), LinearSVC(class_weight="balanced")),
    }


def evaluate(make_model, df, folds):
    scores = {"macro_f1": [], "scam_precision": [], "scam_recall": [], "false_alarm_rate": []}
    for tr, te in folds:
        m = make_model().fit(df["text"].iloc[tr], df["label"].iloc[tr])
        y, p = df["label"].iloc[te].to_numpy(), m.predict(df["text"].iloc[te])
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        scores["macro_f1"].append(f1_score(y, p, average="macro", zero_division=0))
        scores["scam_precision"].append(precision_score(y, p, zero_division=0))
        scores["scam_recall"].append(recall_score(y, p, zero_division=0))
        scores["false_alarm_rate"].append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col"); ap.add_argument("--label-col"); ap.add_argument("--group-col")
    ap.add_argument("--normalized", action="store_true", help="prefer the normalized-text column")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--show-columns", action="store_true")
    ap.add_argument("--out", default="results/baseline.json")
    a = ap.parse_args()

    if a.show_columns:
        raw = data_mod.read_table(a.data)
        print("columns:", list(raw.columns)); print(raw.head(3).to_string()); return

    df = data_mod.load(a.data, a.text_col, a.label_col, a.group_col, a.normalized)
    used = df.attrs["columns_used"]
    print(f"{len(df)} messages | scam={int(df.label.sum())} legit={int((1 - df.label).sum())} | "
          f"templates={df.group.nunique()} | types={df.category.nunique()} | columns used: {used}")
    if used["group"] is None:
        print("WARNING: no template column found; template split == random split. Pass --group-col.")

    split_sets = {"random": splits.random_folds(df, a.folds), "template": splits.template_folds(df, a.folds)}
    if df["category"].nunique() > a.folds:
        split_sets["type"] = splits.category_folds(df, a.folds)
    results = {}
    print(f"\n{'model':<16}{'split':<10}{'macro-F1':>14}{'scam recall':>14}{'false alarms':>14}")
    for name, mk in models().items():
        for split_name, folds in split_sets.items():
            r = evaluate(mk, df, folds)
            results[f"{name}/{split_name}"] = r
            f1, rec, far = r["macro_f1"], r["scam_recall"], r["false_alarm_rate"]
            print(f"{name:<16}{split_name:<10}{f1[0]:>8.3f}±{f1[1]:.3f}{rec[0]:>9.3f}±{rec[1]:.3f}{far[0]:>9.3f}±{far[1]:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"data": a.data, "columns": used, "results": results}, indent=2))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
