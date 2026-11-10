"""Auto-test de bout en bout sur données SYNTHÉTIQUES : aucune donnée, aucun téléchargement, aucun PyTorch.

Fabrique un faux FMA (MP3 via ffmpeg + tracks.csv à deux niveaux d'en-tête) et un faux FakeMusicCaps (WAV),
puis exécute : manifeste -> chaîne commune + codec jouet -> M0 / M1 -> évaluation -> attaques.
`musicdetect selftest` sert à vérifier l'installation (ffmpeg, soxr, scipy, scikit-learn) ; les valeurs d'AUC
n'ont aucune portée scientifique (le « codec » jouet ajoute un peigne de raies synthétique).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .audio.io import require_binary, write_wav
from .config import load_config
from .pipeline import (
    get_paths,
    step_build_fma,
    step_build_fmc,
    step_evaluate,
    step_fmc_index,
    step_manifest,
    step_train,
)


def synth_music(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Signal « musical » : notes harmoniques à enveloppe, plus un fond de bruit rose faible."""
    n = int(seconds * sr)
    x = np.zeros(n)
    pos = 0
    while pos < n:
        dur = int(rng.uniform(0.2, 0.6) * sr)
        f0 = rng.choice([110, 130.8, 164.8, 196, 246.9, 293.7, 392, 440, 523.3])
        t = np.arange(min(dur, n - pos)) / sr
        env = np.sin(np.pi * np.arange(len(t)) / max(len(t), 1)) ** 2
        note = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k ** 1.2 for k in range(1, 20)
                   if f0 * k < sr / 2.1)
        x[pos:pos + len(t)] += env * note
        pos += len(t)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    spec[1:] /= np.sqrt(f[1:] / 50.0)
    pink = np.fft.irfft(spec, n)
    x = x / (np.abs(x).max() + 1e-9) * 0.5 + 0.02 * pink / (pink.std() + 1e-9)
    return (x / (np.abs(x).max() + 1e-9) * 0.8).astype(np.float32)


def make_fake_fma(root: Path, n_tracks: int, n_artists: int, seconds: float, sr: int, rng) -> None:
    """Arborescence `fma_small/000/000002.mp3` + `fma_metadata/tracks.csv` (format FMA)."""
    require_binary("ffmpeg")
    audio, meta = root / "fma_small", root / "fma_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    lines = [",artist,set,set,track,track", ",id,split,subset,genre_top,duration", "track_id,,,,,"]
    for k in range(n_tracks):
        tid = 2 + k
        d = audio / f"{tid:06d}"[:3]
        d.mkdir(parents=True, exist_ok=True)
        wav = d / f"{tid:06d}.wav"
        write_wav(wav, synth_music(seconds, sr, rng), sr, dtype="float32")
        subprocess.run([require_binary("ffmpeg"), "-v", "error", "-y", "-i", str(wav), "-ac", "2",
                        "-c:a", "libmp3lame", "-b:a", "128k", str(d / f"{tid:06d}.mp3")], check=True)
        wav.unlink()
        lines.append(f"{tid},{k % n_artists + 100},training,small,Electronic,{seconds:.0f}")
    (meta / "tracks.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_fake_fmc(root: Path, n_clips: int, rng) -> None:
    """`MusicCaps/*.wav` (réel) et `toygen/*.wav` (faux : peigne jouet), 10 s, 16 kHz, mono."""
    from .codecs.toy import ToyCombCodec
    toy = ToyCombCodec(native_sr=16000, f0=75.0, rel_db=-26.0)
    for k in range(n_clips):
        x = synth_music(10.0, 16000, rng)
        write_wav(root / "MusicCaps" / f"clip{k:03d}.wav", x, 16000)
        write_wav(root / "toygen" / f"clip{k:03d}.wav", toy.reconstruct(x), 16000)


def run_selftest(workdir: str | Path | None = None, n_tracks: int = 24, n_clips: int = 20,
                 verbose: bool = True) -> pd.DataFrame:
    """Exécute tout le pipeline en petit ; renvoie le tableau des résultats (M0 résiduel et M1)."""
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="musicdetect_selftest_")
        workdir = tmp.name
    root = Path(workdir)
    rng = np.random.default_rng(0)
    make_fake_fma(root / "raw" / "fma", n_tracks, 12, 21.0, 22050, rng)
    make_fake_fmc(root / "raw" / "fmc", n_clips, rng)
    try:
        import soxr  # noqa: F401
        backend = "soxr"
    except ImportError:
        backend = "scipy"
        print("[selftest] soxr absent : repli scipy, réservé à cet auto-test.")
    cfg = load_config(overrides=[
        f"paths.fma_root={root / 'raw' / 'fma'}", f"paths.fmc_root={root / 'raw' / 'fmc'}",
        f"paths.work_dir={root / 'work'}", f"paths.results_dir={root / 'results'}",
        f"audio.resample_backend={backend}", "fmc.generators.toygen=[toygen]",
        "eval.bootstrap=200", "m1.C=1.0",
        "eval.attacks.speed=[1.1]", "eval.attacks.speed_extended=[]", "eval.attacks.pitch_semitones=[2]",
        "eval.attacks.mp3_kbps=[128]", "eval.attacks.resample_hz=[11025]", "eval.attacks.noise_snr_db=[20]",
    ])
    cfg.fmc.generators["toygen"] = ["toygen"]
    paths = get_paths(cfg)
    step_manifest(cfg, paths)
    step_fmc_index(cfg, paths)
    step_build_fma(cfg, paths, ["toy_comb_75hz", "identity_24k"])
    step_build_fmc(cfg, paths)
    results = []
    for det, variant in [("m1", None), ("m0", "residual")]:
        step_train(cfg, paths, det, "toy_comb_75hz", variant=variant)
        tests = ["fma:toy_comb_75hz", "fma:identity_24k", "fmc:toygen"]
        results.append(step_evaluate(cfg, paths, det, "toy_comb_75hz", tests, attacks=(det == "m1"),
                                     attack_on="fma:toy_comb_75hz", bootstrap=100, variant=variant))
    out = pd.concat(results, ignore_index=True)
    if verbose:
        print(out[["detector", "train_codec", "test", "auc", "auc_lo", "auc_hi", "eer"]].to_string(index=False))
    if tmp is not None:
        tmp.cleanup()
    return out
