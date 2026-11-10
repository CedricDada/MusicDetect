"""Petits utilitaires communs."""

from __future__ import annotations

try:
    from tqdm import tqdm
except ImportError:  # tqdm est une dépendance déclarée ; ce repli ne sert qu'aux environnements minimaux
    def tqdm(iterable=None, **_kwargs):  # type: ignore[no-redef]
        return iterable
