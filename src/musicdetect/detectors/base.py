"""Interface des détecteurs et extraction de caractéristiques en parallèle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np

from ..data.segments import SegmentSet
from ..utils import tqdm


class Detector(ABC):
    """Un détecteur apprend sur un SegmentSet (étiquettes 0 réel / 1 faux) et renvoie un score de « faux »."""

    name: str = "detector"

    @abstractmethod
    def fit(self, train: SegmentSet, val: SegmentSet | None = None) -> Detector: ...

    @abstractmethod
    def score(self, segs: SegmentSet) -> np.ndarray:
        """Score croissant avec la vraisemblance « faux » (probabilité dans [0, 1] quand c'est possible)."""

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> Detector:
        return joblib.load(path)


def _chunk(segs: SegmentSet, idxs: np.ndarray, fn: Callable[[np.ndarray, int], np.ndarray]) -> np.ndarray:
    return np.stack([fn(*segs.load(int(i))) for i in idxs])


def extract(segs: SegmentSet, fn: Callable[[np.ndarray, int], np.ndarray], n_jobs: int = 1,
            desc: str = "caractéristiques") -> np.ndarray:
    """Applique `fn(x, sr) -> vecteur` à chaque segment ; `fn` doit être picklable si n_jobs > 1."""
    n = len(segs)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n_jobs == 1:
        return np.stack([fn(*segs.load(i)) for i in tqdm(range(n), desc=desc)])
    chunks = np.array_split(np.arange(n), max(n_jobs * 4, 1))
    parts = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_chunk)(segs, c, fn) for c in chunks if len(c))
    return np.concatenate(parts, axis=0)
