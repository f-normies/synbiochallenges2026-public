"""Isotonic calibration: sklearn PAVA at fit time, stored knots for a pure-numpy predict.

sklearn is available in the `esm` env (fit) and the dev `.venv` (tests); the fitted knots are stored
in the probe `.npz` so predict/load use only `np.interp` — no sklearn dependency downstream.
"""

import numpy as np

__all__ = ["fit_isotonic", "apply_isotonic"]


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a non-decreasing isotonic map y~x; return its (x_knots, y_knots)."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip").fit(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    return np.asarray(iso.X_thresholds_, dtype=float), np.asarray(iso.y_thresholds_, dtype=float)


def apply_isotonic(x_knots: np.ndarray, y_knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation on the knots; clamps to the end knots outside the range."""
    return np.interp(np.asarray(x, dtype=float), x_knots, y_knots)
