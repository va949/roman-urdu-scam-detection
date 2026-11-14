"""Train the TF-IDF + LogReg model with a template-aware train/val/test split, then save it.

- Tunes regularisation C and the decision threshold on VALIDATION.
- Scores TEST once per seed, then the robustness suite (perturbed test + hand-written probes).
- --augment K adds K perturbed copies of every train (and 1 of every val) message, using
  only the SEEN perturbation families; UNSEEN families stay test-only.

Usage:
    python src/train.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2
    python src/train.py --data data/raw/phishing_dataset_1000.xlsx --seeds 0 1 2 --augment 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline

import data as data_mod
import perturb
import robustness
import splits
from metrics import full, pick_threshold, ranking, report  # noqa: F401  (re-exported for older imports)


def build(C):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def split_texts(df, seed, augment):
    """Train/val/test texts and labels; train and val get augmented copies when augment > 0."""
    tr, va, te = splits.train_val_test(df, seed=seed)
    X, y = df["text"].tolist(), df["label"].to_numpy()
    Xtr, ytr = [X[i] for i in tr], list(y[tr])
    Xva, yva = [X[i] for i in va], list(y[va])
    if augment:
        at, ay, _ = perturb.augment(Xtr, ytr, k=augment, seed=seed)
        vt, vy, _ = perturb.augment(Xva, yva, k=1, seed=seed + 1000)
        Xtr, ytr, Xva, yva = Xtr + at, ytr + ay, Xva + vt, yva + vy
    return (tr, va, te), (Xtr, np.array(ytr)), (Xva, np.array(yva)), ([X[i] for i in te], y[te])


def run(df, seed, augment):
    (tr, va, te), (Xtr, ytr), (Xva, yva), (Xte, yte) = split_texts(df, seed, augment)
    print(f"\n=== seed {seed}: train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"({'augmented x' + str(augment) if augment else 'no augmentation'}) | templates "
          f"{df.group.iloc[tr].nunique()}/{df.group.iloc[va].nunique()}/{df.group.iloc[te].nunique()}")

    # 1) pick C on validation (ties keep the smallest C, i.e. the strongest regularisation)
    best = None
    for C in (0.3, 1.0, 3.0, 10.0):
        m = build(C).fit(Xtr, ytr)
        f1 = f1_score(yva, m.predict(Xva), average="macro", zero_division=0)
        print(f"  C={C:<5} val macro-F1={f1:.3f}")
        if best is None or f1 > best[1]:
            best = (C, f1, m)
    C, _, model = best

    # 2) pick the decision threshold on validation (trade recall vs false alarms)
    pv = model.predict_proba(Xva)[:, 1]
    thr = pick_threshold(yva, pv)

    # 3) one look at the locked test set, then the robustness suite
    proba = lambda texts: model.predict_proba(list(texts))[:, 1]
    rob = robustness.evaluate(proba, thr, Xte, yte, seed)
    t = rob["clean"]
    print(f"  C={C}, threshold {thr:.3f} | TEST macro-F1 {t['macro_f1']:.3f}  accuracy {t['accuracy']:.3f}  "
          f"recall {t['scam_recall']:.3f}  false alarms {t['false_alarm_rate']:.3f}  ROC-AUC {t['roc_auc']:.3f}")
    result = {"seed": seed, "C": C, "threshold": round(thr, 3), "val": full(yva, pv, thr),
              "test": t, "robustness": rob}
    return model, thr, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col"); ap.add_argument("--label-col"); ap.add_argument("--group-col")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="split seeds (same seed = same split as xlmr.py)")
    ap.add_argument("--augment", type=int, default=0, help="perturbed copies per train message (0 = off)")
    ap.add_argument("--out", default="models/tfidf_lr.joblib", help="model of the FIRST seed is saved here")
    ap.add_argument("--results", default=None, help="default: results/tfidf[_augK].json")
    a = ap.parse_args()

    df = data_mod.load(a.data, a.text_col, a.label_col, a.group_col)
    runs = []
    for i, seed in enumerate(a.seeds):
        model, thr, r = run(df, seed, a.augment)
        runs.append(r)
        if i == 0:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "threshold": float(thr), "meta": r}, a.out)
            print(f"  saved {a.out}")

    print(f"\nTF-IDF{' + augmentation x' + str(a.augment) if a.augment else ''}, mean over seeds {a.seeds}:")
    print(robustness.table([r["robustness"] for r in runs]))
    out = Path(a.results or f"results/tfidf{'_aug' + str(a.augment) if a.augment else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": "tfidf_logreg", "augment": a.augment, "seeds": a.seeds,
                               "summary": robustness.summarize([r["robustness"] for r in runs]),
                               "runs": runs}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
