"""M3 : fakeprint en log-fréquence + corrélation croisée + max-pooling (Dugelay, Barand et al.).

Idée : un changement de vitesse ou de hauteur multiplie toutes les fréquences par un facteur a ; en
log-fréquence cela devient un DÉCALAGE de log2(a) octaves. Le logit est le maximum de la corrélation croisée
entre le fakeprint et un filtre appris : il ne dépend plus du décalage. La perte hybride ajoute, pour les
exemples « faux », une entropie croisée sur la position du maximum (décalage connu de l'attaque).

Référence : `thomas0barand/robust-deepfake-detector` (`RobustDetector`, `LinearProj`), réécrit sans
Lightning ni nnAudio ; la grille log est celle de `stft_to_log` (STFT interpolée), avec F impair.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
from scipy.signal import correlate
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from ..audio.attacks import speed
from ..data.segments import SegmentSet
from ..utils import tqdm
from .base import Detector
from .features import FakeprintConfig, effective_bpo, fakeprint, log_grid, to_log_frequency


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("M3 requiert PyTorch : `uv sync --extra torch`") from exc
    from . import nn_models
    return torch, nn_models


# --------------------------------------------------------------------------- référence numpy


def xcorr_same_numpy(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Corrélation croisée « same » pour F impair : out[i] = sum_j x[i + j - (F-1)/2] w[j] (zéros hors borne).

    Référence de `torch.nn.functional.conv1d(..., padding="same")` ; le pic est en (F-1)/2 + s quand x est w décalé de s.
    """
    F_ = len(w)
    if F_ % 2 == 0:
        raise ValueError("F impair requis")
    left = (F_ - 1) // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), (left, left))
    return correlate(xp, np.asarray(w, dtype=np.float64), mode="valid", method="fft")


def lag_target(factor: float, n_bins: int, bpo_eff: float) -> int:
    """Indice attendu du maximum de la corrélation pour un fakeprint accéléré de `factor`."""
    idx = int(n_bins // 2 + round(float(np.log2(factor)) * bpo_eff))
    return min(max(idx, 0), n_bins - 1)                  # reste dans la fenêtre de corrélation


# --------------------------------------------------------------------------- caractéristiques


def logfp_vector(x: np.ndarray, sr: int, cfg: FakeprintConfig, grid: np.ndarray) -> np.ndarray:
    freqs, fp = fakeprint(x, sr, cfg)
    return to_log_frequency(fp, freqs, grid)


def _aug_chunk(segs: SegmentSet, idxs: np.ndarray, cfg: FakeprintConfig, grid: np.ndarray, factors: np.ndarray,
               backend: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pour chaque segment : version propre (a = 1) puis une variante accélérée par facteur donné."""
    X, y, a = [], [], []
    for row, i in enumerate(idxs):
        x, sr = segs.load(int(i))
        label = int(segs.labels[int(i)])
        for f in [1.0, *factors[row]]:
            xa = x if f == 1.0 else speed(x, sr, float(f), backend)
            X.append(logfp_vector(xa, sr, cfg, grid))
            y.append(label)
            a.append(f)
    return np.stack(X), np.array(y, dtype=np.float32), np.array(a, dtype=np.float32)


class CrossCorrDetector(Detector):
    name = "m3"

    def __init__(self, fp_cfg: FakeprintConfig, cfg: dict, sr: int = 16000, backend: str = "soxr",
                 device: str | None = None, n_jobs: int = 1):
        self.fp_cfg = FakeprintConfig(**{**fp_cfg.__dict__, "normalise": "none"})   # le modèle normalise en L2
        self.cfg = json.loads(json.dumps(dict(cfg)))
        self.sr, self.backend, self.device, self.n_jobs = sr, backend, device, n_jobs
        self.grid = log_grid(self.fp_cfg.band_hz, int(self.cfg["bins_per_octave"]))
        self.bpo_eff = effective_bpo(self.grid)
        self.model = None

    def _dev(self):
        torch, _ = _torch()
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def features(self, segs: SegmentSet) -> np.ndarray:
        from .base import extract
        return extract(segs, partial(logfp_vector, cfg=self.fp_cfg, grid=self.grid), self.n_jobs, desc="M3 fakeprints")

    def _train_matrix(self, train: SegmentSet):
        import joblib
        c = self.cfg
        rng = np.random.default_rng(c["seed"])
        lo, hi = c["speed_range"]
        n_aug = int(c["aug_per_segment"])
        factors = np.exp(rng.uniform(np.log(lo), np.log(hi), size=(len(train), n_aug)))
        chunks = np.array_split(np.arange(len(train)), max(self.n_jobs * 4, 1))
        chunks = [ch for ch in chunks if len(ch)]
        parts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_aug_chunk)(train, ch, self.fp_cfg, self.grid, factors[ch], self.backend)
            for ch in tqdm(chunks, desc="M3 augmentation vitesse"))
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))

    def _logits(self, X: np.ndarray, batch: int = 64):
        torch, _ = _torch()
        dev = self._dev()
        self.model.eval()
        out = []
        with torch.inference_mode():
            for s in range(0, len(X), batch):
                xb = torch.from_numpy(X[s:s + batch]).to(dev)
                logits, _ = self.model(xb, convolve=bool(self.cfg["use_convolution"]))
                out.append(logits.float().cpu().numpy())
        return np.concatenate(out)

    def fit(self, train: SegmentSet, val: SegmentSet | None = None) -> CrossCorrDetector:
        torch, nn_models = _torch()
        import torch.nn.functional as F
        c = self.cfg
        torch.manual_seed(c["seed"])
        dev = self._dev()
        X, y, af = self._train_matrix(train)
        Xv = self.features(val) if val is not None and len(val) else None
        yv = val.labels if Xv is not None else None

        n_bins = X.shape[1]
        self.model = nn_models.XCorrProbe(n_bins).to(dev)
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32, device=dev)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        lag = np.array([lag_target(a, n_bins, self.bpo_eff) for a in af], dtype=np.int64)
        use_conv, lamb, bs = bool(c["use_convolution"]), float(c["lamb"]), int(c["batch_size"])
        rng = np.random.default_rng(c["seed"])

        best, best_state, bad = -np.inf, None, 0
        for epoch in range(int(c["epochs"])):
            self.model.train()
            perm = rng.permutation(len(X))
            run = 0.0
            for s in range(0, len(perm), bs):
                ids = perm[s:s + bs]
                xb = torch.from_numpy(X[ids]).to(dev)
                yb = torch.from_numpy(y[ids]).to(dev)
                logits, cc = self.model(xb, convolve=use_conv)
                loss = bce(logits, yb)
                if use_conv:
                    mask = yb == 1
                    if bool(mask.any()):
                        target = torch.from_numpy(lag[ids]).to(dev)
                        loss = loss + lamb * F.cross_entropy(cc[mask], target[mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                run += float(loss.detach()) * len(ids)
            if Xv is not None:
                metric = roc_auc_score(yv, self._logits(Xv))
            else:
                metric = -run / len(X)
            print(f"[m3] époque {epoch + 1:02d} perte {run / len(X):.4f} métrique {metric:.4f}")
            if metric > best + 1e-4:
                best, bad = metric, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(c["patience"]):
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def score(self, segs: SegmentSet) -> np.ndarray:
        return expit(self._logits(self.features(segs).astype(np.float32)))

    def filter_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(fréquences de la grille, filtre appris) : le motif de pics périodiques doit y apparaître (H1)."""
        return self.grid, self.model.weights.detach().cpu().numpy().ravel()

    def save(self, path) -> None:
        torch, _ = _torch()
        torch.save({"cfg": self.cfg, "fp_cfg": {**self.fp_cfg.__dict__, "band_hz": list(self.fp_cfg.band_hz)},
                    "sr": self.sr, "backend": self.backend,
                    "state": {k: v.cpu() for k, v in self.model.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> CrossCorrDetector:
        torch, nn_models = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=True)
        fp = dict(blob["fp_cfg"])
        fp["band_hz"] = tuple(fp["band_hz"])
        det = cls(FakeprintConfig(**fp), blob["cfg"], blob["sr"], blob["backend"], device)
        det.model = nn_models.XCorrProbe(len(det.grid))
        det.model.load_state_dict(blob["state"])
        det.model.to(det._dev()).eval()
        return det
