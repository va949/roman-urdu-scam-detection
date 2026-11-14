"""Export the TF-IDF + LogReg model for on-device use (the iOS demo in mobile/ios/).

Writes:
- scam_model.json: vocabularies, IDF weights, LogReg coefficients, intercept, threshold
- parity_cases.json: messages + scikit-learn's scam probabilities, for the Swift parity test

Word terms containing 7+ consecutive digits (phone numbers, OTPs, IDs memorised from single
training messages) are pruned from the pipeline first, so no such numbers are published.

Before writing, a pure-Python scorer that mirrors the Swift code step by step (its own
tokenisation, no scikit-learn) is checked against scikit-learn on every parity case.
Parity cases use only synthetic text (probes, perturbed probes, edge cases), so no
Zenodo messages are redistributed.

Usage:
    python src/export_mobile.py --model models/tfidf_lr_aug.joblib
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import perturb

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "mobile" / "ios" / "ScamClassifierKit"
EDGE_CASES = [
    "", "   ", "a", "OK", "Bhai", "!!!", "12345", "Rs.500", "a/c", "J@zzCash 0TP $end", "ALL CAPS MESSAGE HERE",
    "tabs\tand\nnewlines  and   spaces", "under_score words_here", "Aap ka account 🔒 block 🚫 ho gaya",
    "آپ کا اکاؤنٹ بلاک ہو گیا ہے", "Mixed آپ ka اکاؤنٹ block", "Café naïve résumé", "x" * 300,
]


# ---- the reference scorer: keep this in lock-step with ScamClassifier.swift --------------------
def is_word_char(c: str) -> bool:
    """Python's regex \\w for str patterns: alphanumeric (Unicode) or underscore."""
    return c.isalnum() or c == "_"


def char_wb_ngrams(text: str, lo: int, hi: int) -> list[str]:
    """scikit-learn's char_wb analyzer: n-grams inside each space-padded word."""
    out = []
    for w in text.split():
        w = f" {w} "
        for n in range(lo, hi + 1):
            offset = 0
            out.append(w[offset:offset + n])
            while offset + n < len(w):
                offset += 1
                out.append(w[offset:offset + n])
            if offset == 0:  # word shorter than n: counted once, stop growing n
                break
    return out


def word_ngrams(text: str, lo: int, hi: int) -> list[str]:
    """Tokens = maximal runs of >= 2 word characters (token_pattern (?u)\\b\\w\\w+\\b), then n-grams."""
    tokens, cur = [], []
    for c in text:
        if is_word_char(c):
            cur.append(c)
        else:
            if len(cur) >= 2:
                tokens.append("".join(cur))
            cur = []
    if len(cur) >= 2:
        tokens.append("".join(cur))
    out = []
    for n in range(lo, hi + 1):
        out += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return out


def block_score(grams: list[str], block: dict) -> float:
    """Sublinear TF x IDF, L2-normalised within the block, dotted with the LogReg weights."""
    counts: dict[int, int] = {}
    for g in grams:
        idx = block["index"].get(g)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    weights = {i: (1.0 + math.log(c)) * block["idf"][i] for i, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in weights.values()))
    if norm == 0.0:
        return 0.0
    return sum(v / norm * block["coef"][i] for i, v in weights.items())


def reference_proba(text: str, model: dict) -> float:
    t = text.lower()
    z = model["intercept"]
    z += block_score(char_wb_ngrams(t, *model["char"]["ngram_range"]), model["char"])
    z += block_score(word_ngrams(t, *model["word"]["ngram_range"]), model["word"])
    return 1.0 / (1.0 + math.exp(-z))


# ---- export ------------------------------------------------------------------------------------
def export_block(vec, coef):
    terms = [None] * len(vec.vocabulary_)
    for term, i in vec.vocabulary_.items():
        terms[i] = term
    return {"analyzer": vec.analyzer, "ngram_range": list(vec.ngram_range), "terms": terms,
            "idf": [float(x) for x in vec.idf_], "coef": [float(x) for x in coef]}


LONG_DIGITS = re.compile(r"\d{7,}")


def prune_long_numbers(pipe):
    """Copy of the pipeline without word terms containing 7+ consecutive digits.

    Those terms are phone numbers, OTPs and transaction IDs memorised from single training
    messages: they must not be published (some may be real numbers) and they only fire on
    the exact message they came from. Pruning happens inside scikit-learn (vocabulary, idf,
    coefficients), so the exported model and its parity cases stay exactly consistent.
    """
    pipe = copy.deepcopy(pipe)
    fu, lr = pipe.steps[0][1], pipe.steps[1][1]
    (_, cv), (_, wv) = fu.transformer_list
    n_char = len(cv.vocabulary_)
    old_terms = sorted(wv.vocabulary_, key=wv.vocabulary_.get)
    keep = [i for i, t in enumerate(old_terms) if not LONG_DIGITS.search(t)]
    idf = wv.idf_[keep]
    wv.vocabulary_ = {old_terms[i]: j for j, i in enumerate(keep)}
    wv.idf_ = idf
    wv._tfidf.n_features_in_ = len(keep)  # the inner TfidfTransformer validates input width
    lr.coef_ = np.hstack([lr.coef_[:, :n_char], lr.coef_[:, n_char:][:, keep]])
    lr.n_features_in_ = lr.coef_.shape[1]
    return pipe, len(old_terms) - len(keep)


def check_pipeline(pipe):
    fu, lr = pipe.steps[0][1], pipe.steps[1][1]
    names = [n for n, _ in fu.transformer_list]
    assert names == ["char", "word"] and fu.transformer_weights is None, names
    for _, v in fu.transformer_list:
        assert v.lowercase and v.sublinear_tf and v.norm == "l2" and v.use_idf and not v.binary
        assert v.strip_accents is None and v.preprocessor is None and v.tokenizer is None
        assert v.token_pattern == r"(?u)\b\w\w+\b" and v.stop_words is None
    assert fu.transformer_list[0][1].analyzer == "char_wb" and fu.transformer_list[1][1].analyzer == "word"
    assert list(lr.classes_) == [0, 1] and lr.coef_.shape[0] == 1
    return fu, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tfidf_lr_aug.joblib")
    ap.add_argument("--out-model", default=str(KIT / "Sources" / "ScamClassifierKit" / "Resources" / "scam_model.json"))
    ap.add_argument("--out-parity", default=str(KIT / "Tests" / "ScamClassifierKitTests" / "Resources" / "parity_cases.json"))
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    original, thr = bundle["model"], float(bundle["threshold"])
    check_pipeline(original)
    pipe, n_pruned = prune_long_numbers(original)
    fu, lr = check_pipeline(pipe)
    print(f"pruned {n_pruned} word terms containing 7+ digit runs (phone numbers, OTPs, IDs)")
    raw = ROOT / "data" / "raw" / "phishing_dataset_1000.xlsx"
    if raw.exists():  # how much does pruning change predictions on the real data?
        texts = pd.read_excel(raw)["Raw_text"].astype(str).tolist()
        po, pp = original.predict_proba(texts)[:, 1], pipe.predict_proba(texts)[:, 1]
        print(f"  effect on the 1,000 dataset messages: max |diff| {np.max(np.abs(po - pp)):.4f}, "
              f"mean {np.mean(np.abs(po - pp)):.4f}, verdict flips {int(np.sum((po >= thr) != (pp >= thr)))}")
    assert not any(LONG_DIGITS.search(t) for t in fu.transformer_list[1][1].vocabulary_)
    (_, cv), (_, wv) = fu.transformer_list
    n_char = len(cv.vocabulary_)
    model = {
        "format": "tfidf-logreg-v1",
        "threshold": thr,
        "intercept": float(lr.intercept_[0]),
        "char": export_block(cv, lr.coef_[0][:n_char]),
        "word": export_block(wv, lr.coef_[0][n_char:]),
        "meta": {"source_model": Path(a.model).name, "seed": bundle["meta"].get("seed"),
                 "C": bundle["meta"].get("C"), "pruned_long_number_terms": n_pruned,
                 "trained_on": "Roman Urdu Phishing SMS Dataset (G. Siddiq, Zenodo 2026, "
                               "doi:10.5281/zenodo.21810885, CC BY 4.0) + perturbation augmentation"},
    }

    # parity cases: probes, every probe under every perturbation family, edge cases
    probes = pd.read_csv(ROOT / "data" / "probes" / "handwritten_v1.csv")["text"].tolist()
    rng = random.Random(0)
    cases = list(dict.fromkeys(probes + [fn(t, rng) for t in probes for fn in perturb.FAMILIES.values()] + EDGE_CASES))
    sk = pipe.predict_proba(cases)[:, 1]

    lookup = {k: {**model[k], "index": {t: i for i, t in enumerate(model[k]["terms"])}} for k in ("char", "word")}
    ref = np.array([reference_proba(t, {**model, **lookup}) for t in cases])
    worst = float(np.max(np.abs(ref - sk)))
    print(f"reference scorer vs scikit-learn on {len(cases)} cases: max |diff| = {worst:.2e}")
    assert worst < 1e-9, "reference scorer drifted from scikit-learn; fix before exporting"

    for path, obj in ((a.out_model, model), (a.out_parity, {"threshold": thr, "cases": [
            {"text": t, "proba": float(p)} for t, p in zip(cases, sk)]})):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {path} ({Path(path).stat().st_size / 1024:.0f} KB)")
    print(f"features: {n_char} char n-grams + {len(wv.vocabulary_)} word n-grams | threshold {thr:.3f}")


if __name__ == "__main__":
    main()
