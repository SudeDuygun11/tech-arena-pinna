"""Subject-wise K-fold splitting. Both ears of a subject always stay together.

ROLE IN THE PIPELINE
--------------------
Reading order : 5 of 20   (foundations)
Duty          : Subject-wise k-fold splitting.

Splits by SUBJECT, never by ear: left and right of one person are the same
anatomy mirrored, so an ear-wise split would leak between train and test.
"""
from __future__ import annotations

import numpy as np


def k_fold_subject_split(subject_ids: list[str], k: int, seed: int = 0):
    ids = list(subject_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = [ids[i::k] for i in range(k)]
    for i in range(k):
        val_ids = folds[i]
        train_ids = [sid for j, f in enumerate(folds) if j != i for sid in f]
        yield train_ids, val_ids


def single_subject_split(subject_ids: list[str], train_fraction: float, seed: int = 0):
    """One fixed train/test split (not k-fold) at an arbitrary ratio, e.g. 0.85 or 0.90 --
    for ratios that don't correspond to a clean 1/k fraction. Yields a single
    (train_ids, test_ids) pair, matching k_fold_subject_split's interface so
    callers can use either interchangeably."""
    ids = list(subject_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = round(len(ids) * train_fraction)
    yield ids[:n_train], ids[n_train:]
