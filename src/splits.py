"""Two ways to split: random (what most prior work did) and template-aware (honest)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def random_folds(df, k=5, seed=0):
    """Stratified K-fold ignoring templates: near-duplicates can leak into the test set."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(df["text"], df["label"]))


def train_val_test(df, val=0.15, test=0.15, seed=0):
    """Template-aware holdout: ~70/15/15 by template, with BOTH classes in every part.

    Each class's templates are shuffled and dealt out separately: test first, then
    val, each taking templates that fit under its share of that class's messages
    (at least one template each, the smallest if none fit); the rest go to train.
    A mixed-label template counts as its majority label.

    Test is the locked final exam (use once). Tune everything on val.
    """
    rng = np.random.default_rng(seed)
    grp = df["group"].to_numpy()
    sizes = df["group"].value_counts()
    majority = df.groupby("group")["label"].mean().ge(0.5).astype(int)
    parts = {"train": [], "val": [], "test": []}
    for lab in (0, 1):
        groups = majority.index[majority == lab].to_numpy()
        if len(groups) < 3:
            raise ValueError(f"label {lab} has only {len(groups)} template(s); need >= 3 so "
                             "train, val and test each get one. Use template_folds instead.")
        remaining = list(groups[rng.permutation(len(groups))])
        total = sizes[remaining].sum()
        for part, frac in (("test", test), ("val", val)):
            target, taken = frac * total, 0
            reserve = 2 if part == "test" else 1  # leave >= 1 template for each later part
            for g in list(remaining):
                if len(remaining) > reserve and taken + sizes[g] <= target:
                    parts[part].append(g); remaining.remove(g); taken += sizes[g]
            if taken == 0:  # every template overshoots the target: take the smallest
                g = min(remaining, key=lambda x: sizes[x])
                parts[part].append(g); remaining.remove(g)
        parts["train"].extend(remaining)
    tr, va, te = (np.flatnonzero(np.isin(grp, parts[p])) for p in ("train", "val", "test"))
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(grp[a]) & set(grp[b]), "template leaked across split"
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        assert df["label"].iloc[idx].nunique() == 2, f"{name} split has only one class"
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
