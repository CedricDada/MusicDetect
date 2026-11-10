"""M2 : petit CNN sur spectrogramme de magnitude (esprit de `SimpleSpectrogramCNN`, deezer/deepfake-detector)."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from ..data.segments import SegmentSet
from .base import Detector
from .features import power_db_frames


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("M2 requiert PyTorch : `uv sync --extra torch`") from exc
    from . import nn_models
    return torch, nn_models


def spec_db(x: np.ndarray, sr: int, n_fft: int, hop: int, band_hz: tuple[float, float]) -> np.ndarray:
    """Spectrogramme de puissance en dB restreint à la bande ; forme (n_freq, T)."""
    db = power_db_frames(x, n_fft, hop)
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return db[(f >= band_hz[0]) & (f <= band_hz[1])]


class SpecCNNDetector(Detector):
    name = "m2"

    def __init__(self, cfg: dict, sr: int = 16000, device: str | None = None, num_workers: int = 0):
        self.cfg = json.loads(json.dumps(dict(cfg)))        # copie plain-dict (sérialisable)
        self.sr, self.device, self.num_workers = sr, device, num_workers
        self.model = None
        self.mean_, self.std_ = 0.0, 1.0
        self.n_freq: int | None = None

    # ------------------------------------------------------------------ utilitaires
    def _dev(self):
        torch, _ = _torch()
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _to_spec(self, x: np.ndarray, sr: int) -> np.ndarray:
        c = self.cfg
        s = spec_db(x, sr, c["n_fft"], c["hop"], tuple(c["band_hz"]))
        return ((s - self.mean_) / self.std_).astype(np.float32)

    def _build(self, n_freq: int):
        _, nn_models = _torch()
        c = self.cfg
        return nn_models.SpecCNN(n_freq, list(c["hidden"]), list(c["strides"]), c["kernel"],
                                 c["head_dense"], c["dropout"])

    def _predict(self, segs: SegmentSet) -> np.ndarray:
        torch, _ = _torch()
        dev = self._dev()
        self.model.eval()
        out = np.zeros(len(segs), dtype=np.float32)
        bs = int(self.cfg["batch_size"])
        with torch.inference_mode():
            for start in range(0, len(segs), bs):
                idx = list(range(start, min(start + bs, len(segs))))
                specs = [self._to_spec(*segs.load(i)) for i in idx]
                if len({s.shape for s in specs}) == 1:
                    logits = self.model(torch.from_numpy(np.stack(specs)).to(dev))
                    out[idx] = torch.sigmoid(logits).cpu().numpy()
                else:                                               # longueurs différentes (attaque de vitesse)
                    for i, s in zip(idx, specs, strict=True):
                        out[i] = torch.sigmoid(self.model(torch.from_numpy(s)[None].to(dev))).item()
        return out

    # ------------------------------------------------------------------ API
    def fit(self, train: SegmentSet, val: SegmentSet | None = None) -> SpecCNNDetector:
        torch, nn_models = _torch()
        c = self.cfg
        torch.manual_seed(c["seed"])
        np.random.seed(c["seed"])
        dev = self._dev()

        # statistiques de normalisation sur un échantillon de segments d'entraînement
        rng = np.random.default_rng(c["seed"])
        pick = rng.choice(len(train), size=min(200, len(train)), replace=False)
        sample = np.concatenate([spec_db(*train.load(int(i)), c["n_fft"], c["hop"], tuple(c["band_hz"])).ravel()
                                 for i in pick])
        self.mean_, self.std_ = float(sample.mean()), float(sample.std() + 1e-6)
        n_freq = spec_db(np.zeros(c["n_fft"], np.float32), self.sr, c["n_fft"], c["hop"], tuple(c["band_hz"])).shape[0]
        self.n_freq = n_freq
        self.model = self._build(n_freq).to(dev)

        crop = int(round(c["crop_s"] * self.sr)) if c.get("crop_s") else None
        ds = nn_models.WaveSpecDataset(train, self._to_spec, crop, train=True, seed=c["seed"])
        loader = torch.utils.data.DataLoader(ds, batch_size=c["batch_size"], shuffle=True,
                                             num_workers=self.num_workers, drop_last=len(ds) > c["batch_size"])
        y = train.labels
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32, device=dev)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

        best, best_state, bad = -np.inf, None, 0
        for epoch in range(int(c["epochs"])):
            ds.epoch = epoch
            self.model.train()
            run = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                run += float(loss.detach()) * len(yb)
            if val is not None and len(val):
                metric = roc_auc_score(val.labels, self._predict(val))
            else:
                metric = -run / len(ds)                            # sans validation : perte d'entraînement
            print(f"[m2] époque {epoch + 1:02d} perte {run / len(ds):.4f} métrique {metric:.4f}")
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
        return self._predict(segs)

    def save(self, path) -> None:
        torch, _ = _torch()
        torch.save({"cfg": self.cfg, "sr": self.sr, "mean": self.mean_, "std": self.std_,
                    "n_freq": self.n_freq, "state": {k: v.cpu() for k, v in self.model.state_dict().items()}}, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> SpecCNNDetector:
        torch, _ = _torch()
        blob = torch.load(path, map_location="cpu", weights_only=True)
        det = cls(blob["cfg"], blob["sr"], device)
        det.mean_, det.std_, det.n_freq = blob["mean"], blob["std"], blob["n_freq"]
        det.model = det._build(det.n_freq)
        det.model.load_state_dict(blob["state"])
        det.model.to(det._dev()).eval()
        return det
