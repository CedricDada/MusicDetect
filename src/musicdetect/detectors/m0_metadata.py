"""M0 : contrôle négatif, détecteur de raccourcis.

variant = "residual" : descripteurs triviaux calculés sur le segment APRÈS la chaîne de normalisation
                       (rolloff 85/95 %, RMS, silences de début et de fin, plancher de bruit, écrêtage,
                       composante continue). Confirmatoire : si son AUC dépasse 0,5, il reste un raccourci.
variant = "raw"      : métadonnées de fichier (fréquence native, canaux, débit, durée, codec) lues dans
                       les colonnes `meta_*` de l'index. Diagnostic seulement : la séparation FMA contre
                       FakeMusicCaps non normalisés est triviale (MP3 44,1 kHz contre WAV 16 kHz).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.segments import SegmentSet
from .base import Detector, extract
from .features import M0_FEATURES, m0_residual_features

RAW_NUMERIC = ["meta_sample_rate", "meta_channels", "meta_bit_rate", "meta_duration"]


class MetadataDetector(Detector):
    name = "m0"

    def __init__(self, variant: str = "residual", classifier: str = "hgb", silence_db: float = -50.0,
                 seed: int = 0, n_jobs: int = 1):
        if variant not in ("residual", "raw"):
            raise ValueError("variant : residual | raw")
        self.variant, self.classifier_name, self.silence_db = variant, classifier, silence_db
        self.seed, self.n_jobs = seed, n_jobs
        self.codecs_: list[str] = []
        self.clf = None

    # ------------------------------------------------------------------ caractéristiques
    @property
    def feature_names(self) -> list[str]:
        if self.variant == "residual":
            return list(M0_FEATURES)
        return RAW_NUMERIC + [f"codec={c}" for c in self.codecs_]

    def _raw(self, segs: SegmentSet) -> np.ndarray:
        df = segs.df
        num = df[RAW_NUMERIC].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        num = np.nan_to_num(num, nan=-1.0)
        codec = df["meta_codec"].astype(str).to_numpy()
        onehot = np.stack([(codec == c).astype(np.float64) for c in self.codecs_], axis=1) if self.codecs_ \
            else np.zeros((len(df), 0))
        return np.hstack([num, onehot])

    def features(self, segs: SegmentSet) -> np.ndarray:
        if self.variant == "raw":
            return self._raw(segs)
        fn = partial(m0_residual_features, silence_db=self.silence_db)
        return extract(segs, fn, self.n_jobs, desc="M0 résiduel")

    # ------------------------------------------------------------------ API
    def _make_clf(self):
        if self.classifier_name == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        if self.classifier_name == "hgb":
            return HistGradientBoostingClassifier(class_weight="balanced", random_state=self.seed)
        raise ValueError("classifier : logreg | hgb")

    def fit(self, train: SegmentSet, val: SegmentSet | None = None) -> MetadataDetector:
        if self.variant == "raw":
            self.codecs_ = sorted(train.df["meta_codec"].astype(str).unique())
        X = self.features(train)
        self.clf = self._make_clf().fit(X, train.labels)
        return self

    def score(self, segs: SegmentSet) -> np.ndarray:
        return self.clf.predict_proba(self.features(segs))[:, 1]
