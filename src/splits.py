"""Two ways to split: random (what most prior work did) and template-aware (honest)."""
from __future__ import annotations

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def random_folds(df, k=5, seed=0):
    """Stratified K-fold ignoring templates: near-duplicates can leak into the test set."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(df["text"], df["label"]))


def train_val_test(df, val=0.15, test=0.15, seed=0):
    """Template-aware holdout: ~70/15/15 by template, stratified by label.

    Test is the locked final exam (use once). Tune everything on val.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    def carve(idx, frac):
        sub = df.iloc[idx]
        n = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=min(n, sub["group"].nunique()), shuffle=True, random_state=seed)
        keep, out = next(sgkf.split(sub["text"], sub["label"], groups=sub["group"]))
        return idx[keep], idx[out]

    all_idx = np.arange(len(df))
    rest, te = carve(all_idx, test)
    tr, va = carve(rest, val / (1 - test))
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(df["group"].iloc[a]) & set(df["group"].iloc[b]), "template leaked across split"
    return tr, va, te


def category_folds(df, k=5, seed=0):
    """Hold out whole scam/message TYPES: can the model spot a kind of scam it never saw?"""
    tmp = df.assign(group=df["category"])
    return template_folds(tmp, k, seed)


def template_folds(df, k=5, seed=0):
    """Grouped K-fold: every template lands entirely in train or entirely in test."""
    n_groups = df["group"].nunique()
    k = min(k, n_groups)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df["text"], df["label"], groups=df["group"]))
    for tr, te in folds:
        assert not set(df["group"].iloc[tr]) & set(df["group"].iloc[te]), "template leaked across split"
    return folds
