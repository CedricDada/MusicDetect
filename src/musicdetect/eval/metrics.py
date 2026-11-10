"""Métriques de détection : AUC, EER, TPR à FPR fixé avec seuil fixé sur la validation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def eer(y: np.ndarray, s: np.ndarray) -> float:
    """Taux d'égale erreur (moyenne de FPR et FNR au point où ils se croisent)."""
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def threshold_at_fpr(real_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Seuil tel qu'environ `fpr` des scores RÉELS (de validation propre) le dépassent."""
    return float(np.quantile(np.asarray(real_scores, dtype=np.float64), 1.0 - fpr))


def rates_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float]:
    """(TPR, FPR) observés pour le seuil `thr` ; la dérive du FPR sous attaque est le résultat d'intérêt."""
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    tpr = float(np.mean(pos > thr)) if len(pos) else float("nan")
    fpr = float(np.mean(neg > thr)) if len(neg) else float("nan")
    return tpr, fpr
