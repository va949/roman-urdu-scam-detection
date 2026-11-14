"""Print a markdown comparison of results/*.json written by train.py and xlmr.py.

Usage:
    python src/compare.py                       # macro-F1, all runs found
    python src/compare.py --metric false_alarm_rate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = [("TF-IDF", "results/tfidf.json"), ("TF-IDF + aug", "results/tfidf_aug2.json"),
        ("XLM-R", "results/xlmr.json"), ("XLM-R + aug", "results/xlmr_aug2.json")]
ROWS = ["clean", "spelling", "drop_threat", "swap_entity", "currency", "char_tricks", "casing_punct", "probes"]
UNSEEN = {"char_tricks", "casing_punct"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="macro_f1",
                    choices=["macro_f1", "accuracy", "scam_recall", "false_alarm_rate", "roc_auc"])
    a = ap.parse_args()
    found = [(name, json.loads(Path(p).read_text())) for name, p in RUNS if Path(p).exists()]
    found = [(n, j) for n, j in found if "summary" in j and "clean" in j["summary"]]
    if not found:
        raise SystemExit("no results with a robustness summary found; run train.py / xlmr.py first")
    print(f"{a.metric}, mean +/- std over seeds\n")
    print("| Test condition | " + " | ".join(n for n, _ in found) + " |")
    print("|---|" + "---|" * len(found))
    for row in ROWS:
        label = f"{row} (unseen)" if row in UNSEEN else row
        cells = []
        for _, j in found:
            m = j["summary"].get(row, {}).get(a.metric)
            cells.append(f"{m[0]:.3f} +/- {m[1]:.3f}" if m else "n/a")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
