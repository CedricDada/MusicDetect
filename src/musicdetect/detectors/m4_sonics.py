"""M4 : SONICS SpecTTTra (Rahman et al., ICLR 2025), inférence seule.

Poids : `awsaf49/sonics-spectttra-*` sur le Hub, chargés par `sonics.HFAudioClassifier.from_pretrained`
(extra « sonics »). Entrée : 16 kHz mono, recadrages de 5 s normalisés par l'écart-type comme dans
`sonics/utils/dataset.py` ; les recadrages contigus d'un segment sont moyennés (probabilités).
Code non exécuté lors de l'écriture (ni PyTorch ni le paquet `sonics` disponibles).
"""

from __future__ import annotations

import numpy as np

from ..audio.resample import resample
from ..data.segments import SegmentSet
from .base import Detector


class SonicsDetector(Detector):
    name = "m4"

    def __init__(self, hf_id: str = "awsaf49/sonics-spectttra-gamma-5s", sample_rate: int = 16000,
                 crop_s: float = 5.0, batch_size: int = 16, backend: str = "soxr", device: str | None = None):
        self.hf_id, self.sample_rate, self.crop_s = hf_id, sample_rate, crop_s
        self.batch_size, self.backend, self.device = batch_size, backend, device
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                import torch
                from sonics import HFAudioClassifier
            except ImportError as exc:
                raise ImportError("M4 requiert l'extra « sonics » : `uv sync --extra sonics`") from exc
            self._dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = HFAudioClassifier.from_pretrained(self.hf_id).to(self._dev).eval()
        return self._model

    def fit(self, train: SegmentSet, val: SegmentSet | None = None) -> SonicsDetector:
        return self                                   # inférence seule : aucun entraînement

    def _crops(self, x: np.ndarray) -> np.ndarray:
        L = int(round(self.crop_s * self.sample_rate))
        n = max(len(x) // L, 1)
        if len(x) < L:
            x = np.pad(x, (0, L - len(x)))
        crops = np.stack([x[k * L:(k + 1) * L] for k in range(n)]).astype(np.float32)
        return crops / np.maximum(crops.std(axis=1, keepdims=True), 1e-6)

    def score(self, segs: SegmentSet) -> np.ndarray:
        import torch
        model = self._ensure()
        out = np.zeros(len(segs), dtype=np.float32)
        buf, owner = [], []

        def flush():
            if not buf:
                return
            batch = torch.from_numpy(np.concatenate(buf)).to(self._dev)
            with torch.inference_mode():
                p = torch.sigmoid(model(batch).reshape(-1)).cpu().numpy()
            own = np.concatenate(owner)
            for i in np.unique(own):
                out[i] = p[own == i].mean()
            buf.clear()
            owner.clear()

        for i in range(len(segs)):
            x, sr = segs.load(i)
            if sr != self.sample_rate:
                x = resample(x, sr, self.sample_rate, self.backend)
            c = self._crops(x)
            buf.append(c)
            owner.append(np.full(len(c), i))
            if sum(len(b) for b in buf) >= self.batch_size:
                flush()
        flush()
        return out
