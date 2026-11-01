"""Grouped CV helpers."""
import re
import numpy as np
import pandas as pd
import config as cfg


def canon(n):
    return re.sub(r"(bar|\+|-|\^\*|\*|\(S\)|\(L\))", "", n) if isinstance(n, str) else None


def family(r):
    return " ".join(
        sorted(canon(r[p + " Name"]) for p in cfg.P if isinstance(r.get(p + " Name"), str))
    )


def make_folds(groups, seed, k=5):
    groups = np.asarray(groups)
    u, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.argsort(-(cnt + rng.random(len(u)) * 0.9))
    load, fold_of = np.zeros(k), np.zeros(len(u), dtype=int)
    for g in order:
        f = int(np.argmin(load + rng.random(k) * 1e-6))
        fold_of[g] = f
        load[f] += cnt[g]
    return fold_of[inv]


def check_no_group_leakage(groups, fo):
    n_folds = pd.Series(np.asarray(fo)).groupby(np.asarray(groups)).nunique()
    assert (n_folds == 1).all(), f"{int((n_folds > 1).sum())} group(s) straddle folds"


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))
