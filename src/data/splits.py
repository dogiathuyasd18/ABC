"""User-disjoint splits. Windows from one person never cross train/val/test."""

from __future__ import annotations

import random


def split_users(user_ids, val_fraction=0.15, test_fraction=0.15, seed=0):
    """Shuffle users with `seed`, then cut test, val, train.

    Test is held out for a single final report. Hyperparameters may use val only.
    """
    ids = sorted(str(user) for user in user_ids)
    if not ids:
        raise ValueError("No users to split")
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    if n == 1:
        return ids, [], []
    if n == 2:
        return ids[:1], ids[1:], []

    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    if not train_ids:
        train_ids, val_ids = val_ids, []
    _assert_disjoint(train_ids, val_ids, test_ids)
    return train_ids, val_ids, test_ids


def _assert_disjoint(train, val, test) -> None:
    sets = [set(train), set(val), set(test)]
    if len(sets[0] & sets[1]) or len(sets[0] & sets[2]) or len(sets[1] & sets[2]):
        raise RuntimeError("User split leaked across partitions")
