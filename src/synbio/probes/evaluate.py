"""Held-out brightness evaluation by deployment regime + a gate (replaces pooled-Pearson A/B)."""

import numpy as np

from synbio.probes.metrics import pearson, spearman

__all__ = ["regime_report", "regime_gate"]


def regime_report(
    y: np.ndarray, yhat: np.ndarray, edges: tuple[float, float] = (-1.5, -0.7)
) -> dict:
    """Per-regime quality. live=y>edges[0], bright=y>edges[1]; plus pooled (reported only).

    Each regime: {n, pearson, spearman, slope}; pooled: {pearson, spearman}.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    out: dict = {
        "pooled": {"pearson": pearson(y, yhat), "spearman": spearman(y, yhat)}
    }
    for name, lo in (("live", edges[0]), ("bright", edges[1])):
        m = y > lo
        ym, ph = y[m], yhat[m]
        slope = (
            float(np.polyfit(ym, ph, 1)[0])
            if int(m.sum()) > 2
            else float("nan")
        )
        out[name] = {
            "n": int(m.sum()),
            "pearson": pearson(ym, ph),
            "spearman": spearman(ym, ph),
            "slope": slope,
        }
    return out


def regime_gate(report: dict, bright_spearman_min: float = 0.40) -> bool:
    """Pass iff held-out bright-regime Spearman ≥ threshold (the recipe reaches ~0.67)."""
    return report["bright"]["spearman"] >= bright_spearman_min
