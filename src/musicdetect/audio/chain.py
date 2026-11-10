"""Chaîne de normalisation commune (proposition, §chaîne de normalisation commune).

Pour une piste FMA décodée en mono à sa fréquence native :

  1. branche « faux »  : x -> f_c (codec) -> reconstruction -> ancrage
  2. branche « réel »  : x -> f_c -> ancrage, MÊME aller-retour de rééchantillonnage, sans codec
  3. alignement du faux sur le réel (retard estimé par corrélation croisée)
  4. politique de silence identique (rognage décidé sur le réel, appliqué aux deux) ou aucune
  5. segmentation en fenêtres de `segment_s`, mêmes positions de départ (gigue aléatoire) des deux côtés
  6. normalisation de niveau (RMS cible) par segment, identique des deux côtés, avec drapeau d'écrêtage

Les clips externes (FakeMusicCaps) passent par les étapes 4 à 6 seulement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import align_to
from .resample import resample


@dataclass(frozen=True)
class ChainConfig:
    anchor_sr: int = 16000
    segment_s: float = 10.0
    segments_per_track: int = 2
    roundtrip: bool = True
    normalise_level: bool = True
    level_dbfs: float = -23.0
    peak_limit: float = 0.99
    silence_policy: str = "none"          # none | trim
    silence_db: float = -50.0
    backend: str = "soxr"
    max_delay_s: float = 0.05
    seed: int = 0

    @classmethod
    def from_cfg(cls, audio_cfg) -> ChainConfig:
        return cls(
            anchor_sr=int(audio_cfg["anchor_sr"]),
            segment_s=float(audio_cfg["segment_s"]),
            segments_per_track=int(audio_cfg["segments_per_track"]),
            roundtrip=bool(audio_cfg["roundtrip"]),
            normalise_level=bool(audio_cfg["normalise_level"]),
            level_dbfs=float(audio_cfg["level_dbfs"]),
            peak_limit=float(audio_cfg["peak_limit"]),
            silence_policy=str(audio_cfg["silence_policy"]),
            silence_db=float(audio_cfg["silence_db"]),
            backend=str(audio_cfg["resample_backend"]),
            max_delay_s=float(audio_cfg["max_delay_s"]),
            seed=int(audio_cfg["seed"]),
        )

    @property
    def segment_len(self) -> int:
        return int(round(self.segment_s * self.anchor_sr))


@dataclass
class SegmentRecord:
    source: str                 # ex. real_sr24000, encodec_24k_6kbps, fmc:musicgen
    k: int                      # rang du segment dans la piste
    start_s: float
    audio: np.ndarray
    label: int                  # 0 réel, 1 faux
    gain_db: float = 0.0
    limited: bool = False
    delay_samples: int = 0
    lead_silence_s: float = 0.0
    trail_silence_s: float = 0.0
    extra: dict = field(default_factory=dict)


def real_source_name(native_sr: int) -> str:
    return f"real_sr{int(native_sr)}"


# --------------------------------------------------------------------------- briques


def real_branch(x: np.ndarray, sr: int, f_c: int, cfg: ChainConfig) -> np.ndarray:
    """Branche réel : aller-retour x -> f_c -> ancrage (ou x -> ancrage si `roundtrip` est faux)."""
    if cfg.roundtrip:
        return resample(resample(x, sr, f_c, cfg.backend), f_c, cfg.anchor_sr, cfg.backend)
    return resample(x, sr, cfg.anchor_sr, cfg.backend)


def fake_branch(x: np.ndarray, sr: int, codec, cfg: ChainConfig) -> np.ndarray:
    """Branche faux : x -> f_c -> codec -> ancrage."""
    xc = resample(x, sr, codec.native_sr, cfg.backend)
    yc = codec.reconstruct(xc)
    return resample(yc, codec.native_sr, cfg.anchor_sr, cfg.backend)


def silence_bounds(x: np.ndarray, sr: int, thresh_db: float, frame_s: float = 0.02) -> tuple[int, int]:
    """Indices [début, fin) de la partie non silencieuse (trames dont le niveau RMS > seuil dBFS)."""
    n = max(int(frame_s * sr), 1)
    m = len(x) // n
    if m == 0:
        return 0, len(x)
    frames = x[: m * n].reshape(m, n).astype(np.float64)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    active = np.flatnonzero(rms_db > thresh_db)
    if active.size == 0:
        return 0, len(x)
    return int(active[0] * n), int(min((active[-1] + 1) * n, len(x)))


def segment_starts(n: int, seg_len: int, n_seg: int, rng: np.random.Generator) -> list[int]:
    """Départs de segments contigus avec une gigue de départ aléatoire dans la marge disponible."""
    n_fit = min(n_seg, n // seg_len)
    if n_fit <= 0:
        return []
    slack = n - n_fit * seg_len
    offset = int(rng.integers(0, slack + 1))
    return [offset + k * seg_len for k in range(n_fit)]


def normalise_level(seg: np.ndarray, level_dbfs: float, peak_limit: float) -> tuple[np.ndarray, float, bool]:
    """Gain pour atteindre le RMS cible ; si le pic dépasse `peak_limit`, le gain est réduit (limited=True)."""
    rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
    if rms < 1e-8:
        return seg, 0.0, False
    gain = (10.0 ** (level_dbfs / 20.0)) / rms
    peak = float(np.max(np.abs(seg))) * gain
    limited = False
    if peak > peak_limit:
        gain *= peak_limit / peak
        limited = True
    return (seg * gain).astype(np.float32), float(20.0 * np.log10(gain)), limited


# --------------------------------------------------------------------------- pistes complètes


def _cut_and_normalise(sources: dict[str, np.ndarray], labels: dict[str, int], starts: list[int],
                       cfg: ChainConfig, lead_s: float, trail_s: float, lags: dict[str, int],
                       offset: int) -> list[SegmentRecord]:
    out: list[SegmentRecord] = []
    L = cfg.segment_len
    for name, sig in sources.items():
        for k, s in enumerate(starts):
            seg = sig[s:s + L].astype(np.float32)
            gain_db, limited = 0.0, False
            if cfg.normalise_level:
                seg, gain_db, limited = normalise_level(seg, cfg.level_dbfs, cfg.peak_limit)
            out.append(SegmentRecord(
                source=name, k=k, start_s=(offset + s) / cfg.anchor_sr, audio=seg, label=labels[name],
                gain_db=gain_db, limited=limited, delay_samples=lags.get(name, 0),
                lead_silence_s=lead_s, trail_silence_s=trail_s,
            ))
    return out


def process_fma_track(x: np.ndarray, sr: int, codecs: dict, cfg: ChainConfig,
                      track_id: int) -> list[SegmentRecord]:
    """Applique la chaîne complète à une piste FMA ; renvoie les segments réel + faux (un par codec).

    `codecs` : dict nom -> objet codec (attributs `native_sr` et méthode `reconstruct`).
    """
    rng = np.random.default_rng([cfg.seed, int(track_id)])
    reals: dict[int, np.ndarray] = {}
    sources: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    lags: dict[str, int] = {}

    for f_c in sorted({int(c.native_sr) for c in codecs.values()}):
        reals[f_c] = real_branch(x, sr, f_c, cfg)
        sources[real_source_name(f_c)] = reals[f_c]
        labels[real_source_name(f_c)] = 0

    max_lag = int(cfg.max_delay_s * cfg.anchor_sr)
    for name, codec in codecs.items():
        ref = reals[int(codec.native_sr)]
        fk = fake_branch(x, sr, codec, cfg)
        fk, lag = align_to(ref, fk, max_lag)
        sources[name], labels[name], lags[name] = fk, 1, lag

    ref0 = next(iter(reals.values()))
    a, b = silence_bounds(ref0, cfg.anchor_sr, cfg.silence_db)
    lead, trail = a / cfg.anchor_sr, (len(ref0) - b) / cfg.anchor_sr
    offset = 0
    if cfg.silence_policy == "trim":
        sources = {k: v[a:b] for k, v in sources.items()}
        offset = a
    elif cfg.silence_policy != "none":
        raise ValueError(f"silence_policy inconnue : {cfg.silence_policy}")

    n = min(len(v) for v in sources.values())
    sources = {k: v[:n] for k, v in sources.items()}
    starts = segment_starts(n, cfg.segment_len, cfg.segments_per_track, rng)
    return _cut_and_normalise(sources, labels, starts, cfg, lead, trail, lags, offset)


def process_external_clip(x: np.ndarray, sr: int, cfg: ChainConfig, source: str, label: int,
                          clip_key: int = 0) -> list[SegmentRecord]:
    """Étapes 4 à 6 pour un clip externe (FakeMusicCaps) ; rééchantillonnage vers l'ancrage si besoin."""
    rng = np.random.default_rng([cfg.seed, 1, int(clip_key)])
    if sr != cfg.anchor_sr:
        x = resample(x, sr, cfg.anchor_sr, cfg.backend)
    a, b = silence_bounds(x, cfg.anchor_sr, cfg.silence_db)
    lead, trail = a / cfg.anchor_sr, (len(x) - b) / cfg.anchor_sr
    offset = 0
    if cfg.silence_policy == "trim":
        x, offset = x[a:b], a
    elif cfg.silence_policy != "none":
        raise ValueError(f"silence_policy inconnue : {cfg.silence_policy}")
    starts = segment_starts(len(x), cfg.segment_len, 1, rng)
    if not starts:                       # clip plus court qu'un segment : on le garde tel quel
        L = len(x)
        seg, gain_db, limited = x.astype(np.float32), 0.0, False
        if cfg.normalise_level:
            seg, gain_db, limited = normalise_level(seg, cfg.level_dbfs, cfg.peak_limit)
        return [SegmentRecord(source, 0, offset / cfg.anchor_sr, seg, label, gain_db, limited, 0, lead,
                              trail, {"short": True, "len": L})]
    return _cut_and_normalise({source: x}, {source: label}, starts, cfg, lead, trail, {}, offset)
