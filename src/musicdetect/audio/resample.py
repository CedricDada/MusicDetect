"""Rééchantillonnage : soxr (par défaut, VHQ) ou scipy (repli explicite pour les tests)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample(x: np.ndarray, sr_in: float, sr_out: float, backend: str = "soxr",
             quality: str = "VHQ") -> np.ndarray:
    """Rééchantillonne un signal mono de `sr_in` vers `sr_out` (fréquences éventuellement non entières).

    La longueur de sortie vaut round(len(x) * sr_out / sr_in) avec soxr ; le repli scipy peut différer
    de quelques échantillons. Les appelants alignent toujours la sortie sur une longueur de référence.
    """
    x = np.asarray(x, dtype=np.float32)
    if float(sr_in) == float(sr_out):
        return x.copy()
    if backend == "soxr":
        try:
            import soxr
        except ImportError as exc:
            raise ImportError(
                "le paquet `soxr` est requis (backend par défaut). Pour des tests sans soxr, "
                "utiliser audio.resample_backend=scipy ; ne jamais produire de résultats avec ce repli."
            ) from exc
        return soxr.resample(x, sr_in, sr_out, quality=quality).astype(np.float32, copy=False)
    if backend == "scipy":
        frac = Fraction(float(sr_out) / float(sr_in)).limit_denominator(4000)
        y = resample_poly(x.astype(np.float64), frac.numerator, frac.denominator,
                          window=("kaiser", 8.6))
        return y.astype(np.float32)
    raise ValueError(f"backend de rééchantillonnage inconnu : {backend}")


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Coupe ou complète par des zéros (à la fin) pour obtenir exactement `n` échantillons."""
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))
