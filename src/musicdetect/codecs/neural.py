"""Codecs neuronaux via `transformers` (extra « torch ») : EnCodec, DAC, Mimi, EnCodec de MusicGen.

Un seul paquet et des poids sur le Hugging Face Hub. On évite volontairement le paquet
`descript-audio-codec` : il dépend de `descript-audiotools`, qui impose `protobuf<3.20` et rend
la résolution des dépendances fragile sous Python 3.13.

Tout le code de ce module est exécuté en `torch.inference_mode()`. Il n'a pas pu être exécuté lors
de l'écriture (pas de PyTorch dans l'environnement de développement) : voir README, « Ce qui a été testé ».
"""

from __future__ import annotations

import numpy as np

from .base import Codec


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch est requis pour les codecs neuronaux : `uv sync --extra torch`") from exc
    return torch


def _pick_device(device: str | None) -> str:
    torch = _torch()
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _HFCodec(Codec):
    """Squelette : charge le modèle une fois, pad à un multiple de `hop`, reconstruit, recoupe."""

    hop: int = 1

    def __init__(self, hf_id: str, name: str, device: str | None = None):
        self.hf_id, self.name, self.device = hf_id, name, device
        self._model = None

    # --- à définir par les sous-classes
    def _load(self):  # -> modèle HF en mode eval, et met à jour self.native_sr / self.hop
        raise NotImplementedError

    def _forward(self, x_t):  # (1, 1, T) -> tenseur audio reconstruit
        raise NotImplementedError

    def _ensure(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def prepare(self) -> None:
        self._ensure()

    @property
    def sample_rate(self) -> int:
        self._ensure()
        return self.native_sr

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        self._ensure()
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        pad = (-n) % max(int(self.hop), 1)
        xp = np.pad(x, (0, pad)) if pad else x
        x_t = torch.from_numpy(xp)[None, None, :].to(self._dev)
        with torch.inference_mode():
            y = self._forward(x_t)
        y = y.detach().float().reshape(-1).cpu().numpy()
        if len(y) < n:
            y = np.pad(y, (0, n - len(y)))
        return y[:n].astype(np.float32)


class EncodecCodec(_HFCodec):
    """EnCodec 24 kHz mono (facebook/encodec_24khz) ; `bandwidth` en kbit/s parmi 1,5 / 3 / 6 / 12 / 24."""

    def __init__(self, hf_id: str = "facebook/encodec_24khz", bandwidth: float = 6.0,
                 name: str = "encodec", device: str | None = None):
        super().__init__(hf_id, name, device)
        self.bandwidth = float(bandwidth)
        self.native_sr = 24000

    def _load(self):
        from transformers import EncodecModel
        self._dev = _pick_device(self.device)
        model = EncodecModel.from_pretrained(self.hf_id).to(self._dev).eval()
        allowed = list(model.config.target_bandwidths)
        if self.bandwidth not in allowed:
            raise ValueError(f"bandwidth {self.bandwidth} hors de {allowed} pour {self.hf_id}")
        self.native_sr = int(model.config.sampling_rate)
        return model

    def _forward(self, x_t):
        return self._model(x_t, bandwidth=self.bandwidth).audio_values


class MusicgenEncodecCodec(_HFCodec):
    """EnCodec 32 kHz embarqué dans MusicGen (décodeur « oracle » des clips MusicGen). Non testé."""

    def __init__(self, hf_id: str = "facebook/musicgen-small", name: str = "encodec_32k_musicgen",
                 device: str | None = None):
        super().__init__(hf_id, name, device)
        self.native_sr = 32000

    def _load(self):
        from transformers import MusicgenForConditionalGeneration
        self._dev = _pick_device(self.device)
        full = MusicgenForConditionalGeneration.from_pretrained(self.hf_id)
        enc = full.audio_encoder.to(self._dev).eval()
        del full
        self.native_sr = int(enc.config.sampling_rate)
        return enc

    def _forward(self, x_t):
        return self._model(x_t).audio_values


class DacCodec(_HFCodec):
    """DAC 44,1 kHz (descript/dac_44khz) ; `n_quantizers` None = tous (9 codebooks, ~7,7 kbit/s)."""

    def __init__(self, hf_id: str = "descript/dac_44khz", n_quantizers: int | None = None,
                 name: str = "dac", device: str | None = None):
        super().__init__(hf_id, name, device)
        self.n_quantizers = n_quantizers
        self.native_sr = 44100

    def _load(self):
        from transformers import DacModel
        self._dev = _pick_device(self.device)
        model = DacModel.from_pretrained(self.hf_id).to(self._dev).eval()
        self.native_sr = int(model.config.sampling_rate)
        self.hop = int(getattr(model.config, "hop_length", 512))
        return model

    def _forward(self, x_t):
        return self._model(x_t, n_quantizers=self.n_quantizers).audio_values


class MimiCodec(_HFCodec):
    """Mimi (kyutai/mimi), 24 kHz, 12,5 trames/s ; `num_quantizers` = 8 correspond à ~1,1 kbit/s."""

    def __init__(self, hf_id: str = "kyutai/mimi", num_quantizers: int = 8, name: str = "mimi",
                 device: str | None = None):
        super().__init__(hf_id, name, device)
        self.num_quantizers = int(num_quantizers)
        self.native_sr = 24000
        self.hop = 1920

    def _load(self):
        from transformers import MimiModel
        self._dev = _pick_device(self.device)
        model = MimiModel.from_pretrained(self.hf_id).to(self._dev).eval()
        self.native_sr = int(model.config.sampling_rate)
        return model

    def _forward(self, x_t):
        return self._model(x_t, num_quantizers=self.num_quantizers).audio_values
