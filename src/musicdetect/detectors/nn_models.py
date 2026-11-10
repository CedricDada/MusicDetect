"""Modules PyTorch de M2 et M3 (importés seulement si l'extra « torch » est installé).

Ce fichier n'a pas pu être exécuté lors de l'écriture (pas de PyTorch dans l'environnement de
développement) ; `tests/test_torch.py` vérifie les formes et l'équivalence avec la référence numpy de M3.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class SpecCNN(nn.Module):
    """CNN 1D sur spectrogramme : la fréquence est l'axe des canaux, la convolution court sur le temps.

    Architecture de `SimpleSpectrogramCNN` de deezer/deepfake-detector (conv -> batchnorm -> relu,
    pooling global par le maximum, dropout, dense, sortie), en PyTorch et avec moins de canaux.
    """

    def __init__(self, n_freq: int, hidden: list[int], strides: list[int], kernel: int = 3,
                 head_dense: int = 64, dropout: float = 0.2):
        super().__init__()
        assert len(hidden) == len(strides), "hidden et strides doivent avoir la même longueur"
        layers: list[nn.Module] = []
        c_in = n_freq
        for h, s in zip(hidden, strides, strict=True):
            layers += [nn.Conv1d(c_in, h, kernel, stride=s, padding=kernel // 2), nn.BatchNorm1d(h), nn.ReLU()]
            c_in = h
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        if head_dense > 0:
            self.fc = nn.Sequential(nn.Linear(c_in, head_dense), nn.ReLU(), nn.Dropout(dropout))
            c_out = head_dense
        else:
            self.fc, c_out = nn.Identity(), c_in
        self.out = nn.Linear(c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x : (B, n_freq, T) -> logits (B,)
        z = self.conv(x).amax(dim=-1)
        return self.out(self.fc(self.drop(z))).squeeze(-1)


class XCorrProbe(nn.Module):
    """Sonde linéaire de Barand et al. (`LinearProj`) : logit = max_k (x ⋆ w)[k] + b.

    x est écrêté à `max_clamp`, normalisé en L2, puis corrélé (conv1d « same ») au filtre appris w de
    même longueur F. Le maximum de la corrélation rend le détecteur invariant à un décalage en
    log-fréquence (changement de vitesse ou de hauteur). F doit être impair.
    """

    def __init__(self, feature_dim: int, use_norm: bool = True, use_bias: bool = True,
                 init_std: float = 0.02, max_clamp: float = 16.0):
        super().__init__()
        if feature_dim % 2 == 0:
            raise ValueError("feature_dim doit être impair (centre du noyau exact)")
        self.use_norm, self.max_clamp = use_norm, max_clamp
        self.weights = nn.Parameter(torch.randn(1, feature_dim) * init_std)
        self.bias = nn.Parameter(torch.zeros(1)) if use_bias else None

    def forward(self, x: torch.Tensor, convolve: bool = True):
        x = torch.clamp(x, max=self.max_clamp)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)
        if convolve:
            cc = F.conv1d(x.unsqueeze(1), self.weights.unsqueeze(1), padding="same").squeeze(1)   # (B, F)
            logits = cc.max(dim=1).values
        else:
            cc, logits = None, (x @ self.weights.T).squeeze(-1)
        if self.bias is not None:
            logits = logits + self.bias
        return logits, cc


class WaveSpecDataset(Dataset):
    """Segments -> (spectrogramme normalisé (n_freq, T), étiquette). Recadrage aléatoire à l'entraînement."""

    def __init__(self, segs, to_spec, crop_samples: int | None = None, train: bool = False, seed: int = 0):
        self.segs, self.to_spec, self.crop, self.train, self.seed = segs, to_spec, crop_samples, train, seed
        self.epoch = 0
        self.labels = segs.labels

    def __len__(self) -> int:
        return len(self.segs)

    def __getitem__(self, i: int):
        x, sr = self.segs.load(i)
        if self.crop is not None and len(x) > self.crop:
            if self.train:
                start = int(np.random.default_rng([self.seed, self.epoch, i]).integers(0, len(x) - self.crop + 1))
            else:
                start = (len(x) - self.crop) // 2
            x = x[start:start + self.crop]
        spec = self.to_spec(x, sr)
        return torch.from_numpy(np.ascontiguousarray(spec)).float(), torch.tensor(float(self.labels[i]))
