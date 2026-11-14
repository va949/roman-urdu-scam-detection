"""Inference: classify new messages with a saved model.

Usage:
    python src/predict.py --model models/tfidf_lr.joblib "Aap ka JazzCash account block ho gaya hai, foran verify karein"
    python src/predict.py --model models/tfidf_lr.joblib --file messages.txt   # one message per line
"""
import argparse

import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tfidf_lr.joblib")
    ap.add_argument("--file")
    ap.add_argument("messages", nargs="*")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    model, thr = bundle["model"], bundle["threshold"]
    msgs = list(a.messages)
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            msgs += [line.strip() for line in f if line.strip()]
    for msg, p in zip(msgs, model.predict_proba(msgs)[:, 1]):
        verdict = "SCAM" if p >= thr else "ok  "
        print(f"{verdict}  p(scam)={p:.2f}  {msg[:90]}")


if __name__ == "__main__":
    main()
