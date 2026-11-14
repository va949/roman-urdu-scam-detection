"""Recover message templates when a dataset has none.

Scam SMS reuse a template and swap amounts, numbers, names and links. We mask those
slots, then link any two messages whose masked word-shingles overlap above a
Jaccard threshold (union-find, so A~B and B~C puts A, B, C in one template).

Pure Python + stdlib so it runs before scikit-learn is installed.

Usage:
    python src/templates.py data/raw/phishing_dataset_1000.xlsx --threshold 0.6
"""
from __future__ import annotations

import re
from itertools import combinations

URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|pk|net|org|info|xyz|link|ly)\S*)", re.I)
NUM = re.compile(r"[+]?\d[\d,.\-/: ]*\d|\d")


def mask(text: str) -> str:
    t = str(text).lower()
    t = URL.sub(" <url> ", t)
    t = NUM.sub(" <num> ", t)
    t = re.sub(r"[^\w<>]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def shingles(text: str, n: int = 3) -> frozenset:
    toks = mask(text).split()
    if len(toks) < n:
        return frozenset([" ".join(toks)])
    return frozenset(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))


def derive_templates(texts, threshold: float = 0.6, n: int = 3) -> list[str]:
    sh = [shingles(t, n) for t in texts]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Bucket by a few shingles to avoid comparing every pair blindly.
    for i, j in combinations(range(len(texts)), 2):
        a, b = sh[i], sh[j]
        if not a or not b:
            continue
        inter = len(a & b)
        if inter and inter / len(a | b) >= threshold:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    roots = [find(i) for i in range(len(texts))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [f"tpl_{ids[r]:04d}" for r in roots]


if __name__ == "__main__":
    import argparse
    import collections
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--text-col", default="Raw_text")
    ap.add_argument("--label-col", default="Labels")
    a = ap.parse_args()

    import openpyxl
    rows = list(openpyxl.load_workbook(a.path, read_only=True).worksheets[0].iter_rows(values_only=True))
    head, rows = list(rows[0]), rows[1:]
    texts = [r[head.index(a.text_col)] for r in rows]
    labels = [r[head.index(a.label_col)] for r in rows]
    for th in a.threshold:
        tpl = derive_templates(texts, th)
        sizes = collections.Counter(tpl)
        mixed = sum(1 for t in sizes if len({labels[i] for i, x in enumerate(tpl) if x == t}) > 1)
        print(f"threshold={th}: {len(sizes)} templates | largest={max(sizes.values())} | "
              f"singletons={sum(1 for v in sizes.values() if v == 1)} | label-mixed templates={mixed}")
    sys.stdout.flush()
