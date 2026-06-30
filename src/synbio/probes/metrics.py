"""Correlation metrics for probe evaluation (pure numpy/pandas; no scipy)."""

import numpy as np
import pandas as pd

__all__ = ["pearson", "spearman", "per_bin_pearson", "per_group_spearman"]


def pearson(y: np.ndarray, yhat: np.ndarray) -> float:
    """Pearson correlation; 0.0 if either input is constant."""
    y = np.asarray(y, dtype=float) - np.mean(y)
    yhat = np.asarray(yhat, dtype=float) - np.mean(yhat)
    denom = np.sqrt(np.sum(y * y) * np.sum(yhat * yhat))
    return float(np.sum(y * yhat) / denom) if denom > 0 else 0.0


def spearman(y: np.ndarray, yhat: np.ndarray) -> float:
    """Spearman correlation = Pearson on average ranks (ties handled by pandas)."""
    yr = pd.Series(np.asarray(y, dtype=float)).rank().to_numpy()
    yhr = pd.Series(np.asarray(yhat, dtype=float)).rank().to_numpy()
    return pearson(yr, yhr)


def per_bin_pearson(
    y: np.ndarray,
    yhat: np.ndarray,
    nmut: np.ndarray,
    edges: list[int],
    min_count: int = 10,
) -> dict[str, float | None]:
    """Pearson within mutation-count bins (lo, hi]; None if a bin has < min_count.

    Bins are (0, edges[0]], (edges[0], edges[1]], ... keyed "lo-hi".
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    nmut = np.asarray(nmut)
    out: dict[str, float | None] = {}
    lo = 0
    for hi in edges:
        mask = (nmut > lo) & (nmut <= hi)
        key = f"{lo}-{hi}"
        out[key] = pearson(y[mask], yhat[mask]) if int(mask.sum()) >= min_count else None
        lo = hi
    return out


def per_group_spearman(
    y: np.ndarray,
    yhat: np.ndarray,
    groups: np.ndarray,
    min_count: int = 10,
) -> dict[str, float | None]:
    """Spearman within each group label; None if a group has < min_count points.

    Used for per-domain (per-family) ΔΔG evaluation — the deployment-relevant metric
    (ranking mutations within one family), mirroring the paper's per-family reporting.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    groups = np.asarray(groups)
    out: dict[str, float | None] = {}
    for g in pd.unique(groups):
        mask = groups == g
        out[str(g)] = spearman(y[mask], yhat[mask]) if int(mask.sum()) >= min_count else None
    return out
