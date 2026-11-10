"""Entrées / sorties audio.

Le décodage passe par ffmpeg (un seul décodeur pour tous les MP3, comme l'exige la chaîne commune) ;
l'écriture WAV passe par scipy. Aucune dépendance à librosa, soundfile ou torchaudio.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"`{name}` est introuvable. Installez ffmpeg >= 6 avec libmp3lame et "
            "librubberband (voir README, section « Outils système »)."
        )
    return path


def run_ffmpeg(args: list[str]) -> bytes:
    """Exécute ffmpeg (sans stdin) et renvoie stdout ; lève RuntimeError avec stderr en cas d'échec."""
    cmd = [require_binary("ffmpeg"), "-v", "error", "-nostdin", *args]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué ({' '.join(args[:6])} ...) : {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def probe(path: str | Path) -> dict:
    """Métadonnées de fichier (ffprobe) : fréquence, canaux, codec, débit, durée, taille."""
    cmd = [
        require_binary("ffprobe"), "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name,bit_rate,bits_per_sample,duration",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe a échoué sur {path} : {proc.stderr.decode(errors='replace')}")
    info = json.loads(proc.stdout)
    streams = info.get("streams") or []
    if not streams:
        raise RuntimeError(f"aucun flux audio dans {path}")
    s, f = streams[0], info.get("format", {})

    def num(*values, default=float("nan")):
        for v in values:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return default

    return {
        "sample_rate": int(num(s.get("sample_rate"))),
        "channels": int(num(s.get("channels"))),
        "codec": s.get("codec_name", ""),
        "bit_rate": num(s.get("bit_rate"), f.get("bit_rate")),
        "bits_per_sample": num(s.get("bits_per_sample"), default=0.0),
        "duration": num(s.get("duration"), f.get("duration")),
        "size": num(f.get("size")),
    }


def decode_mono(path: str | Path, max_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Décode un fichier audio en flottants mono à sa fréquence native.

    Le passage en mono est la moyenne explicite des canaux (et non la matrice de ffmpeg).
    """
    info = probe(path)
    ch = max(info["channels"], 1)
    args = ["-i", str(path), "-vn"]
    if max_seconds is not None:
        args += ["-t", f"{max_seconds:.6f}"]
    args += ["-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = run_ffmpeg(args)
    x = np.frombuffer(raw, dtype="<f4")
    if x.size % ch:
        x = x[: x.size - x.size % ch]
    x = x.reshape(-1, ch).mean(axis=1) if ch > 1 else x.copy()
    return x.astype(np.float32), info["sample_rate"]


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Lit un WAV (int16/int32/float32/float64/uint8) en flottants mono dans [-1, 1]."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = (data.astype(np.float64) / 2147483648.0).astype(np.float32)
    elif data.dtype == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    else:
        x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x), int(sr)


def write_wav(path: str | Path, x: np.ndarray, sr: int, dtype: str = "int16") -> None:
    """Écrit un WAV mono. `int16` : arrondi sans dither, écrêtage à ±1."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    if dtype == "int16":
        out = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    elif dtype == "float32":
        out = x
    else:
        raise ValueError(f"dtype WAV inconnu : {dtype}")
    wavfile.write(str(path), int(sr), out)
