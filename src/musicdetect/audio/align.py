"""Estimation et compensation du retard entre un signal de référence et sa version traitée."""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from .resample import fit_length


def estimate_delay(ref: np.ndarray, y: np.ndarray, max_lag: int, max_len: int = 64000) -> int:
    """Retard (en échantillons) de `y` par rapport à `ref` : y[n] ~ ref[n - lag].

    Corrélation croisée par FFT sur au plus `max_len` échantillons, lag borné à [-max_lag, +max_lag].
    Un lag positif signifie que `y` est en retard ; on l'annule avec `y[lag:]`.
    """
    n = min(len(ref), len(y), max_len)
    if n < 16:
        return 0
    a = np.asarray(ref[:n], dtype=np.float64)
    b = np.asarray(y[:n], dtype=np.float64)
    if not np.any(a) or not np.any(b):
        return 0
    c = correlate(b, a, mode="full", method="fft")      # c[k] ~ sum_n b[n + lag] a[n], lag = k - (n-1)
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= max_lag
    return int(lags[keep][np.argmax(c[keep])])


def align_to(ref: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, int]:
    """Aligne `y` sur `ref` (retard estimé puis longueur identique). Renvoie (y aligné, lag)."""
    lag = estimate_delay(ref, y, max_lag)
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.pad(y, (-lag, 0))
    return fit_length(y, len(ref)), lag
