"""Inference: classify new messages with a saved model.

Usage:
    python src/predict.py --model models/tfidf_lr.joblib "Aap ka JazzCash account block ho gaya hai, foran verify karein"
    python src/predict.py --model models/tfidf_lr.joblib --file messages.txt   # one message per line
    python src/predict.py --model models/xlmr "..."                            # fine-tuned XLM-R folder
"""
import argparse
import os

import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tfidf_lr.joblib")
    ap.add_argument("--file")
    ap.add_argument("messages", nargs="*")
    a = ap.parse_args()

    msgs = list(a.messages)
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            msgs += [line.strip() for line in f if line.strip()]
    if not msgs:
        ap.error("no messages to classify: pass them as arguments or use --file")

    if os.path.isdir(a.model):  # a fine-tuned transformer saved by xlmr.py
        from xlmr import load_predictor
        predict_proba, thr = load_predictor(a.model)
        probs = predict_proba(msgs)
    else:  # a scikit-learn bundle saved by train.py
        bundle = joblib.load(a.model)
        thr = bundle["threshold"]
        probs = bundle["model"].predict_proba(msgs)[:, 1]
    for msg, p in zip(msgs, probs):
        verdict = "SCAM" if p >= thr else "ok  "
        print(f"{verdict}  p(scam)={p:.2f}  {msg[:90]}")


if __name__ == "__main__":
    main()
