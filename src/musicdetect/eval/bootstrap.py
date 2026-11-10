"""Bootstrap groupé (par artiste pour FMA, par légende pour FakeMusicCaps), apparié, et correction de Holm.

Rééchantillonner les segments ou les pistes sous-estimerait l'incertitude : les segments d'un même
artiste (ou d'une même légende) ne sont pas indépendants.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .metrics import auc


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq, inv = np.unique(groups.astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    return uniq, {i: order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))}


def _replicates(groups: np.ndarray, B: int, seed: int):
    uniq, idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    for _ in range(B):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        yield np.concatenate([idx[k] for k in pick])


def grouped_bootstrap(y: np.ndarray, s: np.ndarray, groups: np.ndarray,
                      stat: Callable[[np.ndarray, np.ndarray], float] = auc,
                      B: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(valeur, borne basse, borne haute) de `stat(y, s)` par bootstrap sur les groupes."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    vals = []
    for i in _replicates(groups, B, seed):
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(stat(y[i], s[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(y, s)), float(lo), float(hi)


def paired_delta(y: np.ndarray, sA: np.ndarray, sB: np.ndarray, groups: np.ndarray,
                 stat: Callable[[np.ndarray, np.ndarray], float] = auc, B: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """ΔAUC = stat(A) - stat(B), IC par bootstrap groupé APPARIÉ (mêmes groupes tirés pour A et B)."""
    y, sA, sB, groups = map(np.asarray, (y, sA, sB, groups))
    d = []
    for i in _replicates(groups, B, seed):
        if len(np.unique(y[i])) < 2:
            continue
        d.append(stat(y[i], sA[i]) - stat(y[i], sB[i]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return {"delta": float(stat(y, sA) - stat(y, sB)), "lo": float(lo), "hi": float(hi),
            "p": min(max(p, 1.0 / (len(d) + 1)), 1.0), "excludes_zero": bool(lo > 0 or hi < 0)}


def holm(pvalues: list[float], alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Correction de Holm-Bonferroni : renvoie (p ajustée, rejet de H0) dans l'ordre d'entrée."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return [(float(a), bool(a <= alpha)) for a in adj]
