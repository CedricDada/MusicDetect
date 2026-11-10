"""Attaques de robustesse (proposition, §attaques), appliquées au signal normalisé à l'ancrage.

vitesse       rééchantillonnage : hauteur ET durée changent (comme l'article de Barand et al.)
hauteur       décalage en demi-tons à durée conservée (Rubber Band via ffmpeg)
mp3           aller-retour MP3 (LAME via ffmpeg), à 44,1 kHz par défaut
rééchant.     aller-retour vers une fréquence plus basse
bruit         bruit blanc gaussien à rapport signal/bruit donné
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .align import align_to
from .io import decode_mono, read_wav, run_ffmpeg, write_wav
from .resample import fit_length, resample


def speed(x: np.ndarray, sr: int, factor: float, backend: str = "soxr") -> np.ndarray:
    """Lecture `factor` fois plus rapide : durée / factor, fréquences x factor."""
    if factor <= 0:
        raise ValueError("facteur de vitesse > 0 requis")
    return resample(x, sr * factor, sr, backend)


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Décalage de hauteur à durée conservée (filtre `rubberband` de ffmpeg)."""
    if semitones == 0:
        return x.copy()
    ratio = 2.0 ** (semitones / 12.0)
    with tempfile.TemporaryDirectory(prefix="md_pitch_") as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        write_wav(src, x, sr, dtype="float32")
        run_ffmpeg(["-i", str(src), "-af", f"rubberband=pitch={ratio:.8f}",
                    "-acodec", "pcm_f32le", str(dst)])
        y, sr_out = read_wav(dst)
    if sr_out != sr:
        y = resample(y, sr_out, sr)
    return fit_length(y, len(x))


def mp3_roundtrip(x: np.ndarray, sr: int, kbps: int, mp3_sr: int = 44100,
                  backend: str = "soxr") -> np.ndarray:
    """x -> mp3_sr -> MP3 (LAME, `kbps`, mono) -> décodage -> sr ; retard d'encodeur compensé.

    À 16 kHz l'encodage direct n'autorise pas 192 ou 320 kbit/s (MPEG-2 couche III) : on passe donc
    par `mp3_sr` (44,1 kHz par défaut), comme une piste réellement distribuée.
    """
    up = resample(x, sr, mp3_sr, backend)
    with tempfile.TemporaryDirectory(prefix="md_mp3_") as d:
        src, mp3 = Path(d) / "in.wav", Path(d) / "out.mp3"
        write_wav(src, up, mp3_sr, dtype="float32")
        run_ffmpeg(["-i", str(src), "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{int(kbps)}k", str(mp3)])
        dec, sr_dec = decode_mono(mp3)
    down = resample(dec, sr_dec, sr, backend)
    down, _ = align_to(x, down, max_lag=int(0.05 * sr))
    return down


def resample_roundtrip(x: np.ndarray, sr: int, low_sr: int, backend: str = "soxr") -> np.ndarray:
    """sr -> low_sr -> sr : supprime tout ce qui dépasse low_sr / 2."""
    down = resample(x, sr, low_sr, backend)
    return fit_length(resample(down, low_sr, sr, backend), len(x))


def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Bruit blanc gaussien ; puissance du bruit = puissance du signal / 10^(snr/10)."""
    p = float(np.mean(np.square(x, dtype=np.float64)))
    if p <= 0:
        return x.copy()
    sigma = np.sqrt(p / (10.0 ** (snr_db / 10.0)))
    return (x + rng.normal(0.0, sigma, size=x.shape)).astype(np.float32)


def attack_name(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "speed":
        return f"speed_{spec['factor']:g}"
    if kind == "pitch":
        return f"pitch_{spec['semitones']:+g}"
    if kind == "mp3":
        return f"mp3_{int(spec['kbps'])}"
    if kind == "resample":
        return f"resample_{int(spec['low_sr'])}"
    if kind == "noise":
        return f"noise_{spec['snr_db']:g}"
    if kind == "none":
        return "clean"
    raise ValueError(f"attaque inconnue : {kind}")


def apply_attack(x: np.ndarray, sr: int, spec: dict[str, Any], rng: np.random.Generator | None = None,
                 backend: str = "soxr") -> np.ndarray:
    kind = spec["kind"]
    if kind == "none":
        return x
    if kind == "speed":
        return speed(x, sr, spec["factor"], backend)
    if kind == "pitch":
        return pitch_shift(x, sr, spec["semitones"])
    if kind == "mp3":
        return mp3_roundtrip(x, sr, spec["kbps"], spec.get("mp3_sr", 44100), backend)
    if kind == "resample":
        return resample_roundtrip(x, sr, spec["low_sr"], backend)
    if kind == "noise":
        return add_noise(x, spec["snr_db"], rng or np.random.default_rng(0))
    raise ValueError(f"attaque inconnue : {kind}")


class AttackTransform:
    """Transformation appelable par SegmentSet : (x, sr, index) -> x attaqué (aléa fixé par l'indice)."""

    def __init__(self, spec: dict[str, Any], seed: int = 0, backend: str = "soxr"):
        self.spec, self.seed, self.backend = spec, seed, backend
        self.name = attack_name(spec)

    def __call__(self, x: np.ndarray, sr: int, index: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, index])
        return apply_attack(x, sr, self.spec, rng, self.backend)


def default_attack_grid(cfg_attacks: dict[str, Any]) -> list[dict[str, Any]]:
    """Grille d'attaques de la proposition à partir de `eval.attacks` de la configuration."""
    grid: list[dict[str, Any]] = []
    for f in list(cfg_attacks.get("speed", [])) + list(cfg_attacks.get("speed_extended", [])):
        grid.append({"kind": "speed", "factor": float(f)})
    for s in cfg_attacks.get("pitch_semitones", []):
        grid.append({"kind": "pitch", "semitones": float(s)})
    for k in cfg_attacks.get("mp3_kbps", []):
        grid.append({"kind": "mp3", "kbps": int(k)})
    for r in cfg_attacks.get("resample_hz", []):
        grid.append({"kind": "resample", "low_sr": int(r)})
    for s in cfg_attacks.get("noise_snr_db", []):
        grid.append({"kind": "noise", "snr_db": float(s)})
    return grid
