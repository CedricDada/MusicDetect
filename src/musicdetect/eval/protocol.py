"""Protocole d'évaluation : résumés avec IC groupés, seuil fixé sur la validation, grille d'attaques."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..audio.attacks import AttackTransform, attack_name
from ..data.segments import SegmentSet, combine, make_pair_set, read_index
from ..detectors.base import Detector
from ..utils import tqdm
from .bootstrap import grouped_bootstrap
from .calibration import ece
from .metrics import auc, eer, rates_at_threshold, threshold_at_fpr


def summarise(y: np.ndarray, s: np.ndarray, groups: np.ndarray, thr: float | None = None, B: int = 1000,
              seed: int = 0) -> dict:
    """AUC (IC groupé 95 %), EER, TPR / FPR au seuil fixé, ECE ; `above_chance` : IC de l'AUC exclut 0,5."""
    y, s = np.asarray(y), np.asarray(s)
    a, lo, hi = grouped_bootstrap(y, s, groups, auc, B, seed)
    out = {"n": int(len(y)), "n_fake": int(y.sum()), "n_groups": int(len(np.unique(groups))),
           "auc": a, "auc_lo": lo, "auc_hi": hi, "above_chance": bool(lo > 0.5), "eer": eer(y, s),
           "ece": ece(y, s) if (s.min() >= 0 and s.max() <= 1) else float("nan")}
    if thr is not None:
        out["threshold"] = float(thr)
        out["tpr"], out["fpr"] = rates_at_threshold(y, s, thr)
    return out


def validation_threshold(detector: Detector, val: SegmentSet, fpr: float = 0.01) -> float:
    """Seuil du FPR cible sur les segments RÉELS de validation propre (jamais réajusté sous attaque)."""
    scores = detector.score(val)
    return threshold_at_fpr(scores[val.labels == 0], fpr)


def evaluate_set(detector: Detector, segs: SegmentSet, thr: float | None, B: int = 1000, seed: int = 0,
                 extra: dict | None = None) -> tuple[dict, np.ndarray]:
    s = detector.score(segs)
    row = summarise(segs.labels, s, segs.groups, thr, B, seed)
    row.update(extra or {})
    return row, s


def evaluate_attacks(detector: Detector, segs: SegmentSet, grid: list[dict], thr: float | None,
                     B: int = 1000, seed: int = 0, backend: str = "soxr") -> pd.DataFrame:
    """Une ligne par attaque (+ la ligne « clean ») : AUC avec IC, TPR / FPR au seuil de validation propre."""
    rows = []
    for spec in tqdm([{"kind": "none"}, *grid], desc="attaques"):
        t = None if spec["kind"] == "none" else AttackTransform(spec, seed, backend)
        row, _ = evaluate_set(detector, segs.with_transform(t), thr, B, seed, {"attack": attack_name(spec)})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- résolution des jeux de test


def resolve_test_set(spec: str, fma_index: str | Path, fma_dir: str | Path, fmc_index: str | Path | None = None,
                     fmc_dir: str | Path | None = None, real_source: str | None = None) -> SegmentSet:
    """`fma:<codec>[:val|:test|:train]` : réel apparié contre faux du codec (split `test` par défaut).
    `fmc:<générateur>` : clips générés contre clips réels MusicCaps (source `fmc:real`) s'ils existent ;
    sinon contre le réel FMA du split test : séparation alors confondue avec la fuite de format, à signaler.
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "fma":
        codec = parts[1]
        split = parts[2] if len(parts) > 2 else "test"
        idx = read_index(fma_index)
        if codec not in set(idx["source"]):
            raise KeyError(f"codec {codec!r} absent de l'index FMA ; sources : {sorted(set(idx['source']))}")
        return make_pair_set(idx, fma_dir, split, codec, real_source_of(idx, codec), name=spec)
    if kind == "fmc":
        if fmc_index is None or fmc_dir is None:
            raise ValueError("fmc_index et fmc_dir requis pour un jeu fmc:<générateur>")
        gen = parts[1]
        idx = read_index(fmc_index)
        fake = SegmentSet(idx[idx["source"] == f"fmc:{gen}"], fmc_dir, None, f"fmc:{gen}")
        real_rows = idx[idx["source"] == "fmc:real"]
        if len(real_rows):
            real = SegmentSet(real_rows, fmc_dir, None, "fmc:real")
        else:
            fidx = read_index(fma_index)
            src = real_source or sorted(s for s in set(fidx["source"]) if s.startswith("real_sr"))[0]
            real = SegmentSet(fidx[(fidx["split"] == "test") & (fidx["source"] == src)], fma_dir, None, f"fma:{src}")
        if len(fake) == 0:
            raise KeyError(f"aucun clip du générateur {gen!r} dans {fmc_index}")
        return combine([real, fake], name=spec)
    raise ValueError(f"jeu de test inconnu : {spec}")


def real_source_of(idx: pd.DataFrame, codec: str) -> str:
    """Source réelle appariée à un codec : celle dont les `pair_id` recouvrent ceux du faux."""
    fake_pairs = set(idx.loc[idx["source"] == codec, "pair_id"])
    best, best_n = None, -1
    for src in sorted(s for s in set(idx["source"]) if s.startswith("real_sr")):
        n = len(fake_pairs & set(idx.loc[idx["source"] == src, "pair_id"]))
        if n > best_n:
            best, best_n = src, n
    if best is None:
        raise KeyError("aucune source réelle (real_sr*) dans l'index")
    return best
