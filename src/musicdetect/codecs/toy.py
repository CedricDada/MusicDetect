"""Codecs « jouets » en numpy pur : pour les tests et les démonstrations sans PyTorch.

`ToyCombCodec` ajoute un peigne de raies à n·f0 : c'est la signature que prédit l'explication de Fourier
(pics espacés de la fréquence de trame f_e). Ce n'est PAS un modèle d'EnCodec : il sert à vérifier que la
chaîne, M1 et l'évaluation fonctionnent de bout en bout.
"""

from __future__ import annotations

import numpy as np

from .base import Codec


class IdentityCodec(Codec):
    """Contrôle négatif : ne fait rien (le réel et le faux ne diffèrent que par le bruit de calcul)."""

    def __init__(self, native_sr: int = 24000, name: str = "identity"):
        self.native_sr, self.name = int(native_sr), name

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32).copy()


class ToyCombCodec(Codec):
    """x + peigne de raies à n·f0 jusqu'à Nyquist.

    rel_db : puissance TOTALE du peigne relative à la puissance du signal (dB).
    """

    def __init__(self, native_sr: int = 24000, f0: float = 75.0, rel_db: float = -26.0,
                 f_min: float = 500.0, seed: int = 0, name: str = "toy_comb"):
        self.native_sr, self.f0, self.rel_db = int(native_sr), float(f0), float(rel_db)
        self.f_min, self.seed, self.name = float(f_min), int(seed), name

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        sr = self.native_sr
        n_lo = int(np.ceil(self.f_min / self.f0))
        n_hi = int(np.floor((0.49 * sr) / self.f0))
        ns = np.arange(n_lo, n_hi + 1)
        rng = np.random.default_rng(self.seed)
        phases = rng.uniform(0, 2 * np.pi, size=ns.size)
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) or 1e-3
        # puissance totale du peigne = rms^2 * 10^(rel/10) ; N raies d'amplitude a : N a^2 / 2
        a = rms * np.sqrt(2.0 * 10.0 ** (self.rel_db / 10.0) / ns.size)
        period = sr / self.f0
        if abs(period - round(period)) < 1e-9:       # peigne exactement périodique : une période, répétée
            p = int(round(period))
            t = np.arange(p, dtype=np.float64) / sr
            one = np.sin(2 * np.pi * self.f0 * ns[:, None] * t[None, :] + phases[:, None]).sum(axis=0)
            comb = np.resize(one, len(x))
        else:
            comb = np.zeros(len(x), dtype=np.float64)
            t = np.arange(len(x), dtype=np.float64) / sr
            for i in range(0, ns.size, 32):           # par blocs pour limiter la mémoire
                comb += np.sin(2 * np.pi * self.f0 * ns[i:i + 32, None] * t[None, :]
                               + phases[i:i + 32, None]).sum(axis=0)
        return (x + a * comb).astype(np.float32)
