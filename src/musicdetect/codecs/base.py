"""Interface commune des codecs : reconstruire un signal mono à la fréquence native du codec."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Codec(ABC):
    """Un codec « encode puis décode » : `reconstruct` renvoie un signal de même longueur et même fréquence.

    Les rééchantillonnages d'entrée / de sortie sont faits par la chaîne (audio/chain.py), pas ici.
    """

    name: str = "codec"
    native_sr: int = 24000

    @abstractmethod
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """x : float32 mono à `native_sr`, valeurs dans [-1, 1] -> reconstruction de même longueur."""

    def prepare(self) -> None:  # noqa: B027  (point d'extension facultatif, pas une méthode abstraite)
        """Charge les poids si nécessaire (met à jour `native_sr`) ; sans effet pour les codecs numpy."""

    def describe(self) -> dict:
        return {"name": self.name, "native_sr": self.native_sr}
