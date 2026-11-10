"""Détecteurs M0 à M4 et fabrique pilotée par la configuration."""

from __future__ import annotations

from pathlib import Path

from .base import Detector
from .features import FakeprintConfig
from .m0_metadata import MetadataDetector
from .m1_fakeprint import FakeprintDetector

DETECTORS = ("m0", "m1", "m2", "m3", "m4")


def build_detector(name: str, cfg, device: str | None = None, n_jobs: int = 1, variant: str | None = None) -> Detector:
    """Instancie un détecteur d'après la configuration complète `cfg`."""
    sr = int(cfg.audio.anchor_sr)
    fp = FakeprintConfig.from_cfg(cfg.features)
    if name == "m0":
        return MetadataDetector(variant or cfg.m0.variant, cfg.m0.classifier, cfg.audio.silence_db,
                                cfg.audio.seed, n_jobs)
    if name == "m1":
        return FakeprintDetector(fp, cfg.m1.C, cfg.m1.class_weight, cfg.m1.max_iter,
                                 tune_C=(0.01, 0.1, 1.0, 10.0), n_jobs=n_jobs, sr=sr)
    if name == "m2":
        from .m2_cnn import SpecCNNDetector
        return SpecCNNDetector(dict(cfg.m2), sr, device)
    if name == "m3":
        from .m3_crosscorr import CrossCorrDetector
        return CrossCorrDetector(fp, dict(cfg.m3), sr, cfg.audio.resample_backend, device, n_jobs)
    if name == "m4":
        from .m4_sonics import SonicsDetector
        m = cfg.m4
        return SonicsDetector(m.hf_id, m.sample_rate, m.crop_s, m.batch_size, cfg.audio.resample_backend, device)
    raise ValueError(f"détecteur inconnu : {name} (attendu : {DETECTORS})")


def load_detector(name: str, path: str | Path, device: str | None = None) -> Detector:
    if name == "m2":
        from .m2_cnn import SpecCNNDetector
        return SpecCNNDetector.load(path, device)
    if name == "m3":
        from .m3_crosscorr import CrossCorrDetector
        return CrossCorrDetector.load(path, device)
    return Detector.load(path)            # m0, m1 : joblib


__all__ = ["DETECTORS", "Detector", "FakeprintConfig", "build_detector", "load_detector"]
