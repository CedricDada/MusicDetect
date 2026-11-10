"""Caractéristiques spectrales : fakeprint (M1, M3), spectrogramme (M2), descripteurs triviaux (M0).

Recette du fakeprint (Afchar et al., ISMIR 2025 ; `deezer/ismir25-ai-music-detector`, vectorisée comme
`thomas0barand/robust-deepfake-detector`) :

  1. spectrogramme de puissance en dB (fenêtre de Hann, n_fft, centré), borné à [1e-10, 1e6] avant le dB
  2. moyenne temporelle (des dB par défaut, comme Deezer)
  3. restriction à la bande utile, plancher en dB
  4. enveloppe inférieure : minima locaux sur une fenêtre de `hull_area` bins, interpolés linéairement
  5. résidu = max(spectre - enveloppe, 0) ; normalisation (max comme Deezer / L2 / aucune)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft
from scipy.signal import get_window


@dataclass(frozen=True)
class FakeprintConfig:
    n_fft: int = 8192
    hop: int = 4096
    band_hz: tuple[float, float] = (1000.0, 8000.0)
    floor_db: float = -50.0
    hull_area: int = 20
    average: str = "db"            # db | power
    normalise: str = "max"         # max | l2 | none
    max_db: float = 5.0

    @classmethod
    def from_cfg(cls, features_cfg, **overrides) -> FakeprintConfig:
        d = dict(features_cfg)
        d["band_hz"] = tuple(d["band_hz"])
        d.update(overrides)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# --------------------------------------------------------------------------- STFT


def power_db_frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Spectrogramme de puissance en dB, forme (n_bins, n_frames).

    Équivalent de `torchaudio.transforms.Spectrogram(n_fft, hop, power=2)` (Hann périodique, centré par
    réflexion) suivi de 10 log10(clip(., 1e-10, 1e6)), comme dans le code de Deezer.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    frames = sliding_window_view(xp, n_fft)[::hop]
    win = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    spec = rfft(frames * win, axis=1)
    power = np.clip(spec.real ** 2 + spec.imag ** 2, 1e-10, 1e6)
    return (10.0 * np.log10(power)).T


def mean_spectrum_db(x: np.ndarray, n_fft: int, hop: int, average: str = "db") -> np.ndarray:
    db = power_db_frames(x, n_fft, hop)
    if average == "db":
        return db.mean(axis=1)
    if average == "power":
        return 10.0 * np.log10(np.mean(10.0 ** (db / 10.0), axis=1))
    raise ValueError(f"average inconnu : {average}")


# --------------------------------------------------------------------------- fakeprint


def lower_hull(y: np.ndarray, area: int = 20) -> np.ndarray:
    """Enveloppe inférieure : minima locaux (fenêtre `area`) + extrémités, interpolés linéairement."""
    n = len(y)
    if n <= area:
        return np.full(n, float(np.min(y)))
    windows = sliding_window_view(y, area)
    anchors = windows.argmin(axis=1) + np.arange(windows.shape[0])
    idx = np.unique(np.concatenate([anchors, [0, n - 1]]))
    return np.interp(np.arange(n), idx, y[idx])


def band_freqs(sr: int, n_fft: int, band_hz: tuple[float, float]) -> np.ndarray:
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return f[(f >= band_hz[0]) & (f <= min(band_hz[1], sr / 2))]


def fakeprint(x: np.ndarray, sr: int, cfg: FakeprintConfig) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (fréquences de la bande en Hz, fakeprint) pour un signal mono."""
    spec = mean_spectrum_db(x, cfg.n_fft, cfg.hop, cfg.average)
    f = np.fft.rfftfreq(cfg.n_fft, 1.0 / sr)
    sel = (f >= cfg.band_hz[0]) & (f <= min(cfg.band_hz[1], sr / 2))
    s = np.maximum(spec[sel], cfg.floor_db)
    resid = np.clip(s - lower_hull(s, cfg.hull_area), 0.0, None)
    if cfg.normalise == "max":
        r = np.clip(resid, 0.0, cfg.max_db)
        resid = r / (1e-6 + r.max())
    elif cfg.normalise == "l2":
        resid = resid / (np.linalg.norm(resid) + 1e-9)
    elif cfg.normalise != "none":
        raise ValueError(f"normalise inconnu : {cfg.normalise}")
    return f[sel], resid.astype(np.float32)


def fakeprint_vector(x: np.ndarray, sr: int, cfg: FakeprintConfig) -> np.ndarray:
    return fakeprint(x, sr, cfg)[1]


# --------------------------------------------------------------------------- axe log-fréquence (M3)


def log_grid(band_hz: tuple[float, float], bins_per_octave: int) -> np.ndarray:
    """Grille log-fréquence de taille IMPAIRE (pour que le centre du noyau soit exact)."""
    n = int(np.ceil(bins_per_octave * np.log2(band_hz[1] / band_hz[0])))
    n += 1 - (n % 2)
    return np.geomspace(band_hz[0], band_hz[1], n)


def effective_bpo(grid: np.ndarray) -> float:
    """Nombre réel de bins par octave de la grille (sert à convertir un facteur de vitesse en décalage)."""
    return (len(grid) - 1) / np.log2(grid[-1] / grid[0])


def to_log_frequency(fp: np.ndarray, freqs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolation linéaire du fakeprint (axe linéaire) vers la grille log-fréquence."""
    return np.interp(grid, freqs, fp).astype(np.float32)


# --------------------------------------------------------------------------- M0 : descripteurs triviaux

M0_FEATURES = [
    "rolloff85_hz", "rolloff95_hz", "rms_dbfs", "peak", "crest_db", "lead_silence_ratio",
    "trail_silence_ratio", "noise_floor_db", "clip_ratio", "dc_offset",
]


def m0_residual_features(x: np.ndarray, sr: int, silence_db: float = -50.0, frame_s: float = 0.02) -> np.ndarray:
    """Descripteurs triviaux d'un segment déjà normalisé (« M0 résiduel »)."""
    x = np.asarray(x, dtype=np.float64)
    p = power_db_frames(x.astype(np.float32), 2048, 1024)
    mean_pow = (10.0 ** (p / 10.0)).mean(axis=1)
    cum = np.cumsum(mean_pow) / (mean_pow.sum() + 1e-30)
    f = np.fft.rfftfreq(2048, 1.0 / sr)
    r85, r95 = f[min(np.searchsorted(cum, 0.85), len(f) - 1)], f[min(np.searchsorted(cum, 0.95), len(f) - 1)]
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    n = max(int(frame_s * sr), 1)
    m = max(len(x) // n, 1)
    fr = x[: m * n].reshape(m, n) if len(x) >= n else x[None, :]
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-12)
    active = np.flatnonzero(frame_db > silence_db)
    lead = active[0] / len(frame_db) if active.size else 1.0
    trail = (len(frame_db) - 1 - active[-1]) / len(frame_db) if active.size else 1.0
    return np.array([
        r85, r95, 20.0 * np.log10(rms), peak, 20.0 * np.log10(peak / rms), lead, trail,
        np.percentile(frame_db, 5), float(np.mean(np.abs(x) >= 0.999)), float(np.mean(x)),
    ], dtype=np.float32)
