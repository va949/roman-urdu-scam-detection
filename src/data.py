"""Load a scam-SMS dataset (CSV or XLSX) into a standard frame: text, label (0/1), group.

The Zenodo file's exact column names aren't documented, so columns are detected
by name. Override with --text-col / --label-col / --group-col if detection is wrong.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SCAM_WORDS = {"phishing", "scam", "spam", "fraud", "smishing", "1", "true", "yes"}
LEGIT_WORDS = {"legitimate", "legit", "ham", "normal", "genuine", "0", "false", "no"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _find(cols: list[str], *needles: str, avoid: tuple[str, ...] = ()) -> str | None:
    for needle in needles:
        for c in cols:
            lc = c.lower()
            if needle in lc and not any(a in lc for a in avoid):
                return c
    return None


def detect_columns(df: pd.DataFrame, text_col=None, label_col=None, group_col=None, prefer_normalized=False):
    cols = [str(c) for c in df.columns]
    if text_col is None:
        if prefer_normalized:
            text_col = _find(cols, "normal", "clean")
        text_col = text_col or _find(cols, "raw", "message", "text", "sms", avoid=("id", "normal", "clean", "type"))
        text_col = text_col or _find(cols, "message", "text", "sms", avoid=("id",))
    label_col = label_col or _find(cols, "label", "class", "category", "target", avoid=("source",))
    group_col = group_col or _find(cols, "template", "group", "cluster")
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns in {cols}; pass --text-col/--label-col.")
    return text_col, label_col, group_col


def to_binary(series: pd.Series) -> pd.Series:
    def conv(v):
        s = str(v).strip().lower()
        if s in SCAM_WORDS:
            return 1
        if s in LEGIT_WORDS:
            return 0
        raise ValueError(f"Unknown label value {v!r}; extend SCAM_WORDS/LEGIT_WORDS in data.py")
    return series.map(conv).astype(int)


def load(path, text_col=None, label_col=None, group_col=None, prefer_normalized=False,
         derive_threshold: float | None = 0.6) -> pd.DataFrame:
    """Columns out: text, label, group (template), category (scam type, if the data has one).

    The Zenodo file has no template column, so templates are derived from text
    similarity (see templates.py) unless --group-col names a real one.
    """
    raw = read_table(path)
    t, l, g = detect_columns(raw, text_col, label_col, group_col, prefer_normalized)
    df = pd.DataFrame({"text": raw[t].astype(str), "label": to_binary(raw[l])})
    cat = _find([str(c) for c in raw.columns], "source_type", "category", "type", avoid=("id",))
    df["category"] = raw[cat].astype(str) if cat else "unknown"
    if g is None and derive_threshold is not None:
        from templates import derive_templates
        df["group"] = derive_templates(df["text"].tolist(), derive_threshold)
        df.attrs["columns_used"] = {"text": t, "label": l, "group": f"derived@{derive_threshold}", "category": cat}
        return df.reset_index(drop=True)
    if g is not None:
        groups = raw[g].astype("string")
        # Messages without a template ID each form their own group.
        missing = groups.isna() | (groups.str.strip() == "")
        groups[missing] = [f"_solo_{i}" for i in df.index[missing]]
        df["group"] = groups.astype(str)
    else:
        df["group"] = [f"_solo_{i}" for i in df.index]
    df.attrs["columns_used"] = {"text": t, "label": l, "group": g, "category": cat}
    return df.dropna(subset=["text"]).reset_index(drop=True)
