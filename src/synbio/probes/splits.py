"""Mutation-count-stratified train/test split with a type-aware Hamming-1 guard."""

import numpy as np
import pandas as pd

__all__ = ["mutation_count_stratified_split", "filter_hamming1"]


def _build_train_indices(
    df: pd.DataFrame, train_idx: set[int]
) -> tuple[dict[str, set[frozenset]], dict[str, set[frozenset]]]:
    """Per-type (train mutation sets, all size-(k-1) subsets of train sets)."""
    sets: dict[str, set[frozenset]] = {}
    minus1: dict[str, set[frozenset]] = {}
    tcol = df.columns.get_loc("gfp_type")
    mcol = df.columns.get_loc("mutset")
    for i in train_idx:
        t = df.iat[i, tcol]
        s = df.iat[i, mcol]
        sets.setdefault(t, set()).add(s)
        sub = minus1.setdefault(t, set())
        for el in s:
            sub.add(s - {el})
    return sets, minus1


def _conflicts(s: frozenset, t: str, sets, minus1) -> bool:
    """True if some same-type train set has symmetric-difference size 1 with s."""
    if s in minus1.get(t, set()):  # train T = s + one element
        return True
    type_sets = sets.get(t, set())
    for el in s:  # train T = s - one element
        if (s - {el}) in type_sets:
            return True
    return False


def filter_hamming1(
    df: pd.DataFrame, train_idx: list[int], test_idx: list[int]
) -> list[int]:
    """Return the test indices with NO same-type train set at symmetric-difference 1."""
    sets, minus1 = _build_train_indices(df, set(train_idx))
    tcol = df.columns.get_loc("gfp_type")
    mcol = df.columns.get_loc("mutset")
    return [
        i for i in test_idx
        if not _conflicts(df.iat[i, mcol], df.iat[i, tcol], sets, minus1)
    ]


def mutation_count_stratified_split(
    df: pd.DataFrame,
    test_frac: float,
    seed: int,
    hamming1_guard: bool = True,
) -> tuple[list[int], list[int]]:
    """Split df rows (needs columns `gfp_type`, `mutset`) into train/test index lists.

    Test is sampled at `test_frac` within each mutation-count stratum. With the guard,
    any test row whose mutation set differs from a same-type train set by exactly one
    element is moved back to train (removes near-duplicate memorization).
    """
    rng = np.random.default_rng(seed)
    nmut = df["mutset"].apply(len).to_numpy()
    test_idx: set[int] = set()
    for k in np.unique(nmut):
        members = np.where(nmut == k)[0]
        rng.shuffle(members)
        n_test = int(round(len(members) * test_frac))
        test_idx.update(int(i) for i in members[:n_test])

    if hamming1_guard:
        train_idx = sorted(set(range(len(df))) - test_idx)
        test_idx = set(filter_hamming1(df, train_idx, sorted(test_idx)))

    test = sorted(test_idx)
    train = sorted(set(range(len(df))) - test_idx)
    return train, test
