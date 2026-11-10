"""M1 : fakeprint + régression logistique (Afchar et al., ISMIR 2025 ; `train_test_regressor.py`)."""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..data.segments import SegmentSet
from .base import Detector, extract
from .features import FakeprintConfig, band_freqs, fakeprint_vector


class FakeprintDetector(Detector):
    name = "m1"

    def __init__(self, fp_cfg: FakeprintConfig, C: float = 1.0, class_weight: str | None = "balanced",
                 max_iter: int = 2000, tune_C: tuple[float, ...] | None = None, n_jobs: int = 1,
                 sr: int = 16000):
        self.fp_cfg, self.C, self.class_weight, self.max_iter = fp_cfg, C, class_weight, max_iter
        self.tune_C, self.n_jobs, self.sr = tune_C, n_jobs, sr
        self.clf: LogisticRegression | None = None

    @property
    def freqs(self) -> np.ndarray:
        return band_freqs(self.sr, self.fp_cfg.n_fft, self.fp_cfg.band_hz)

    def features(self, segs: SegmentSet) -> np.ndarray:
        return extract(segs, partial(fakeprint_vector, cfg=self.fp_cfg), self.n_jobs, desc="fakeprints")

    def _lr(self, C: float) -> LogisticRegression:
        return LogisticRegression(C=C, class_weight=self.class_weight, max_iter=self.max_iter)

    def fit(self, train: SegmentSet, val: SegmentSet | None = None) -> FakeprintDetector:
        X, y = self.features(train), train.labels
        C = self.C
        if self.tune_C and val is not None and len(val):
            Xv, yv = self.features(val), val.labels
            best = max(self.tune_C, key=lambda c: roc_auc_score(yv, self._lr(c).fit(X, y).decision_function(Xv)))
            C = best
        self.C = C
        self.clf = self._lr(C).fit(X, y)
        return self

    def score(self, segs: SegmentSet) -> np.ndarray:
        return self.clf.predict_proba(self.features(segs))[:, 1]

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(fréquences, coefficients) : à comparer aux multiples de la fréquence de trame du codec (H1)."""
        return self.freqs, self.clf.coef_.ravel()
