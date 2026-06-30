"""Merge the three generator fragments into the carrier candidates.parquet.

Exact intra-merge dedup (keep-first, fragments passed in priority order) + exact
Exclusion_List filtering. Fuzzy / Foldseek / FPbase dedup stays in stage 09.
"""

from pathlib import Path

import pandas as pd

from synbio.io.artifacts import validate_candidates

__all__ = ["read_exclusion", "merge_fragments"]


def read_exclusion(path: str | Path) -> set[str]:
    """Load the `Sequence` column of Exclusion_List.csv into a set."""
    return set(pd.read_csv(path)["Sequence"].astype(str))


def merge_fragments(
    frames: list[pd.DataFrame], exclusion: set[str]
) -> tuple[pd.DataFrame, dict]:
    """Concat fragments (union columns), exact keep-first dedup, exact exclusion filter."""
    combined = pd.concat(frames, ignore_index=True)
    n_in = len(combined)
    deduped = combined.drop_duplicates(subset="sequence", keep="first")
    n_dup = n_in - len(deduped)
    excluded_mask = deduped["sequence"].isin(exclusion)
    n_excluded = int(excluded_mask.sum())
    out = deduped[~excluded_mask].reset_index(drop=True)
    validate_candidates(out)
    stats = {"n_in": n_in, "n_dup": n_dup, "n_excluded": n_excluded, "n_out": len(out)}
    return out, stats
