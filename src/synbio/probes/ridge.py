"""Closed-form ridge probe on frozen embeddings (pure numpy)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synbio.probes.isotonic import apply_isotonic, fit_isotonic

__all__ = ["RidgeProbe", "fit_ridge", "CalibratedRidge", "fit_calibrated_ridge"]


@dataclass
class RidgeProbe:
    """A standardized linear ridge model: predict = ((X - mean)/std) @ w + b."""

    w: np.ndarray
    b: float
    x_mean: np.ndarray
    x_std: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.x_mean) / self.x_std
        return Xs @ self.w + self.b

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w=self.w,
            b=np.array(self.b, dtype=float),
            x_mean=self.x_mean,
            x_std=self.x_std,
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeProbe":
        d = np.load(path, allow_pickle=False)
        return cls(
            w=d["w"],
            b=float(d["b"]),
            x_mean=d["x_mean"],
            x_std=d["x_std"],
            meta=json.loads(str(d["meta"].item())),
        )


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sample_weight: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
) -> RidgeProbe:
    """Fit standardized ridge in fp64: w = (Xsᵀ W Xs + αI)⁻¹ Xsᵀ W (y - ȳ_w).

    `sample_weight` (normalized to mean 1 so `alpha` stays comparable) down-weights rows;
    None ⇒ uniform, identical to the unweighted fit.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if sample_weight is None:
        w_s = np.ones(len(y))
    else:
        w_s = np.asarray(sample_weight, dtype=float)
        w_s = w_s / w_s.sum() * len(w_s)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0.0] = 1.0  # guard constant features
    Xs = (X - x_mean) / x_std
    b = float(np.average(y, weights=w_s))
    yc = y - b
    d = Xs.shape[1]
    Xw = Xs * w_s[:, None]
    a_mat = Xw.T @ Xs + alpha * np.eye(d)
    w = np.linalg.solve(a_mat, Xw.T @ yc)
    return RidgeProbe(w=w, b=b, x_mean=x_mean, x_std=x_std, meta=meta or {})


@dataclass
class CalibratedRidge:
    """Dead-weighted ridge + full-range isotonic calibration.

    Pure-numpy predict (np.interp).
    """

    ridge: RidgeProbe
    iso_x: np.ndarray
    iso_y: np.ndarray
    wlo: float
    whi: float
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.clip(self.ridge.predict(X), self.wlo, self.whi)
        return apply_isotonic(self.iso_x, self.iso_y, raw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w=self.ridge.w,
            b=np.array(self.ridge.b, dtype=float),
            x_mean=self.ridge.x_mean,
            x_std=self.ridge.x_std,
            iso_x=self.iso_x,
            iso_y=self.iso_y,
            wlo=np.array(self.wlo, dtype=float),
            whi=np.array(self.whi, dtype=float),
            meta=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedRidge":
        d = np.load(path, allow_pickle=False)
        ridge = RidgeProbe(
            w=d["w"],
            b=float(d["b"]),
            x_mean=d["x_mean"],
            x_std=d["x_std"],
        )
        return cls(
            ridge=ridge,
            iso_x=d["iso_x"],
            iso_y=d["iso_y"],
            wlo=float(d["wlo"]),
            whi=float(d["whi"]),
            meta=json.loads(str(d["meta"].item())),
        )


def fit_calibrated_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    dead_weight: float,
    live_lo: float,
    meta: dict[str, Any] | None = None,
) -> CalibratedRidge:
    """Down-weight dead (y ≤ live_lo) in the ridge; winsorize; isotonic on full y.

    Dead rows are down-weighted (not removed) in the ridge fit.
    Ridge output is clipped to the train-y [0.5%, 99.5%] quantiles (winsorization).
    Isotonic calibration is fit on the FULL y (functional and dead), not live-only —
    this allows dead signal to map low.
    """
    y = np.asarray(y, dtype=float)
    sw = np.where(y > live_lo, 1.0, dead_weight)
    ridge = fit_ridge(X, y, alpha, sample_weight=sw)
    wlo, whi = (float(v) for v in np.quantile(y, [0.005, 0.995]))
    raw = np.clip(ridge.predict(X), wlo, whi)
    iso_x, iso_y = fit_isotonic(raw, y)
    return CalibratedRidge(
        ridge=ridge,
        iso_x=iso_x,
        iso_y=iso_y,
        wlo=wlo,
        whi=whi,
        meta=meta or {},
    )
