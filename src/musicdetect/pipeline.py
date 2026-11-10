"""Étapes du pipeline (appelées par la CLI, par `selftest` et par les tests).

work_dir/
  manifest_fma.csv            pistes FMA retenues + split
  fmc_files.csv               clips FakeMusicCaps découverts
  segments/fma[_tag]/<source>/<piste>_<k>.wav     réel apparié + faux par codec
  index_fma[_tag].csv
  segments/fmc[_tag]/...      clips FakeMusicCaps passés par la chaîne
  index_fmc[_tag].csv
  models/<tag|main>/<détecteur>__<codec>.{joblib,pt,json}
results_dir/<tag|main>/       résultats, scores par segment, attaques
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .audio.attacks import default_attack_grid
from .audio.chain import ChainConfig
from .codecs import build_codecs
from .data import fma as fma_mod
from .data import fmc as fmc_mod
from .data.build import build_fma_segments, build_fmc_segments
from .data.segments import make_pair_set, read_index
from .detectors import build_detector, load_detector
from .eval.bootstrap import paired_delta
from .eval.protocol import (
    evaluate_attacks,
    evaluate_set,
    real_source_of,
    resolve_test_set,
    validation_threshold,
)


@dataclass
class Paths:
    manifest: Path
    fmc_files: Path
    fma_dir: Path
    fma_index: Path
    fmc_dir: Path
    fmc_index: Path
    models: Path
    results: Path


def get_paths(cfg, tag: str | None = None) -> Paths:
    w, r = Path(cfg.paths.work_dir), Path(cfg.paths.results_dir)
    suf = f"_{tag}" if tag else ""
    return Paths(
        manifest=w / "manifest_fma.csv", fmc_files=w / "fmc_files.csv",
        fma_dir=w / "segments" / f"fma{suf}", fma_index=w / f"index_fma{suf}.csv",
        fmc_dir=w / "segments" / f"fmc{suf}", fmc_index=w / f"index_fmc{suf}.csv",
        models=w / "models" / (tag or "main"), results=r / (tag or "main"),
    )


def resolve_variant(cfg, detector: str, variant: str | None) -> str | None:
    """M0 a deux variantes ; on fige celle de la configuration pour que le nom du modèle soit sans ambiguïté."""
    return (variant or cfg.m0.variant) if detector == "m0" else variant


def model_stem(detector: str, codec: str | None, variant: str | None = None) -> str:
    return f"{detector}{'-' + variant if variant else ''}__{codec or 'none'}"


# --------------------------------------------------------------------------- données


def step_manifest(cfg, paths: Paths) -> pd.DataFrame:
    f = cfg.fma
    df = fma_mod.make_manifest(cfg.paths.fma_root, f.subset, f.split_mode, f.val_frac, f.test_frac,
                               cfg.audio.seed, f.max_tracks)
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.manifest, index=False)
    print(f"[manifest] {len(df)} pistes, {df['artist_id'].nunique()} artistes -> {paths.manifest}")
    print(df["split"].value_counts().to_string())
    return df


def step_fmc_index(cfg, paths: Paths) -> pd.DataFrame:
    df = fmc_mod.discover(cfg.paths.fmc_root, {k: list(v) for k, v in cfg.fmc.generators.items()},
                          cfg.fmc.get("group_regex"))
    paths.fmc_files.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.fmc_files, index=False)
    print(fmc_mod.summary(df))
    return df


def step_build_fma(cfg, paths: Paths, codec_names: list[str], device: str | None = None,
                   resume: bool = True) -> pd.DataFrame:
    manifest = pd.read_csv(paths.manifest, dtype={"path": str})
    codecs = build_codecs({k: dict(v) for k, v in cfg.codecs.items()}, codec_names, device)
    return build_fma_segments(manifest, codecs, ChainConfig.from_cfg(cfg.audio), paths.fma_dir, paths.fma_index,
                              cfg.audio.max_track_s, cfg.audio.wav_dtype, resume)


def step_build_fmc(cfg, paths: Paths) -> pd.DataFrame:
    files = pd.read_csv(paths.fmc_files, dtype=str)
    files["label"] = files["label"].astype(int)
    return build_fmc_segments(files, cfg.paths.fmc_root, ChainConfig.from_cfg(cfg.audio), paths.fmc_dir,
                              paths.fmc_index, cfg.audio.wav_dtype)


# --------------------------------------------------------------------------- entraînement / évaluation


def _pair_sets(paths: Paths, codec: str):
    idx = read_index(paths.fma_index)
    real = real_source_of(idx, codec)
    return (make_pair_set(idx, paths.fma_dir, "train", codec, real, f"train:{codec}"),
            make_pair_set(idx, paths.fma_dir, "val", codec, real, f"val:{codec}"), real)


def _model_path(paths: Paths, stem: str, detector: str) -> Path:
    return paths.models / f"{stem}.{'pt' if detector in ('m2', 'm3') else 'joblib'}"


def step_train(cfg, paths: Paths, detector: str, codec: str, device: str | None = None, n_jobs: int = 1,
               variant: str | None = None) -> Path | None:
    """Entraîne `detector` sur (réel apparié, faux du codec), valide, fixe le seuil de FPR cible sur la validation."""
    variant = resolve_variant(cfg, detector, variant)
    train, val, _ = _pair_sets(paths, codec)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(f"jeux train/val vides pour {codec} : vérifier l'index {paths.fma_index}")
    det = build_detector(detector, cfg, device, n_jobs, variant)
    print(f"[train] {detector} sur {train} ; validation {val}")
    det.fit(train, val)
    if detector == "m4":
        print("[train] M4 est en inférence seule : rien à enregistrer.")
        return None
    stem = model_stem(detector, codec, variant)
    path = _model_path(paths, stem, detector)
    path.parent.mkdir(parents=True, exist_ok=True)
    det.save(path)
    s = det.score(val)
    from .eval.metrics import auc, threshold_at_fpr
    thr = threshold_at_fpr(s[val.labels == 0], cfg.eval.fpr_target)
    meta = {"detector": detector, "variant": variant, "codec": codec, "threshold": thr,
            "fpr_target": cfg.eval.fpr_target, "val_auc": auc(val.labels, s), "n_train": len(train),
            "n_val": len(val)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] modèle -> {path} ; AUC validation {meta['val_auc']:.4f} ; seuil {thr:.4f}")
    return path


def step_evaluate(cfg, paths: Paths, detector: str, codec: str, tests: list[str], attacks: bool = False,
                  attack_on: str | None = None, bootstrap: int | None = None, device: str | None = None,
                  n_jobs: int = 1, variant: str | None = None, save_scores: bool = True) -> pd.DataFrame:
    """Évalue un détecteur entraîné sur `codec` : AUC avec IC groupé, EER, TPR/FPR au seuil de validation propre."""
    B = int(bootstrap or cfg.eval.bootstrap)
    variant = resolve_variant(cfg, detector, variant)
    stem = model_stem(detector, codec, variant)
    if detector == "m4":
        det = build_detector("m4", cfg, device, n_jobs)
        thr = None
    else:
        det = load_detector(detector, _model_path(paths, stem, detector), device)
        thr = None
    meta_path = _model_path(paths, stem, detector).with_suffix(".json")
    if meta_path.exists():
        thr = json.loads(meta_path.read_text(encoding="utf-8"))["threshold"]
    else:
        _, val, _ = _pair_sets(paths, codec)
        thr = validation_threshold(det, val, cfg.eval.fpr_target)

    rows = []
    paths.results.mkdir(parents=True, exist_ok=True)
    real_src = real_source_of(read_index(paths.fma_index), codec) if paths.fma_index.exists() else None
    for spec in tests:
        segs = resolve_test_set(spec, paths.fma_index, paths.fma_dir, paths.fmc_index, paths.fmc_dir, real_src)
        row, s = evaluate_set(det, segs, thr, B, cfg.eval.seed,
                              {"detector": detector, "variant": variant or "", "train_codec": codec, "test": spec})
        rows.append(row)
        print(f"[eval] {spec:<28s} AUC {row['auc']:.3f} [{row['auc_lo']:.3f}, {row['auc_hi']:.3f}] "
              f"EER {row['eer']:.3f} TPR@seuil {row.get('tpr', float('nan')):.3f} FPR {row.get('fpr', float('nan')):.3f}")
        if save_scores:
            d = paths.results / "scores"
            d.mkdir(exist_ok=True)
            out = segs.df[["seg_id", "label", "group"]].copy()
            out["score"] = s
            out.to_csv(d / f"{stem}__{spec.replace(':', '_')}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(paths.results / f"{stem}.csv", index=False)

    if attacks:
        spec = attack_on or tests[0]
        segs = resolve_test_set(spec, paths.fma_index, paths.fma_dir, paths.fmc_index, paths.fmc_dir, real_src)
        grid = default_attack_grid(dict(cfg.eval.attacks))
        att = evaluate_attacks(det, segs, grid, thr, B, cfg.eval.seed, cfg.audio.resample_backend)
        att.insert(0, "test", spec)
        att.insert(0, "train_codec", codec)
        att.insert(0, "detector", detector)
        att.to_csv(paths.results / f"attacks_{stem}.csv", index=False)
        print(att[["attack", "auc", "auc_lo", "auc_hi", "tpr", "fpr"]].to_string(index=False))
    return res


def step_compare(scores_a: str | Path, scores_b: str | Path, B: int = 1000, seed: int = 0) -> dict:
    """ΔAUC apparié (A - B) par bootstrap groupé, à partir de deux fichiers de scores par segment."""
    a, b = pd.read_csv(scores_a, dtype={"seg_id": str, "group": str}), pd.read_csv(scores_b, dtype={"seg_id": str, "group": str})
    m = a.merge(b, on="seg_id", suffixes=("_a", "_b"))
    if len(m) == 0:
        raise ValueError("aucun segment commun entre les deux fichiers de scores")
    if not (m["label_a"] == m["label_b"]).all():
        raise ValueError("étiquettes différentes pour un même seg_id")
    return paired_delta(m["label_a"].to_numpy(), m["score_a"].to_numpy(), m["score_b"].to_numpy(),
                        m["group_a"].to_numpy(), B=B, seed=seed) | {"n": int(len(m))}


def collect_results(results_dir: str | Path) -> pd.DataFrame:
    """Concatène tous les CSV de résultats (hors attaques et scores) d'un dossier en un tableau."""
    d = Path(results_dir)
    frames = [pd.read_csv(p) for p in sorted(d.glob("*.csv")) if not p.name.startswith("attacks_")
              and p.name != "summary.csv"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ = ["Paths", "collect_results", "get_paths", "model_stem", "resolve_variant", "step_build_fma", "step_build_fmc",
           "step_compare", "step_evaluate", "step_fmc_index", "step_manifest", "step_train"]
