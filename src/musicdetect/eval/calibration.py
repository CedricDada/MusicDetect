"""Calibration : ECE et mise à l'échelle par température."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error de la probabilité « faux » (bins d'égale largeur)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = which == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def fit_temperature(y: np.ndarray, p: np.ndarray) -> float:
    """Température T > 0 minimisant la log-vraisemblance négative de sigmoid(logit(p) / T) (sur la validation)."""
    z, y = _logit(p), np.asarray(y, dtype=float)

    def nll(logT: float) -> float:
        q = 1.0 / (1.0 + np.exp(-z / np.exp(logT)))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-_logit(p) / T))
