"""Index de segments et accès aux formes d'onde.

Un index est un CSV (une ligne par segment WAV). `SegmentSet` lit les WAV à la demande et applique
éventuellement une transformation (attaque) déterministe par indice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from ..audio.io import read_wav

INDEX_COLUMNS = [
    "seg_id", "path", "label", "source", "track_id", "group", "split", "pair_id", "start_s", "sr",
    "gain_db", "limited", "delay_samples", "lead_silence_s", "trail_silence_s",
    "meta_sample_rate", "meta_channels", "meta_bit_rate", "meta_duration", "meta_codec",
]

Transform = Callable[[np.ndarray, int, int], np.ndarray]


def read_index(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"seg_id": str, "path": str, "source": str, "group": str, "pair_id": str,
                                  "split": str, "meta_codec": str})
    for col in INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def write_index(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in INDEX_COLUMNS if c in df.columns] + [c for c in df.columns if c not in INDEX_COLUMNS]
    df[cols].to_csv(path, index=False)


class SegmentSet:
    """Ensemble de segments : étiquettes, groupes, et lecture paresseuse des formes d'onde."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform: Transform | None = None, name: str = ""):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy(dtype=int)

    @property
    def groups(self) -> np.ndarray:
        return self.df["group"].astype(str).to_numpy()

    def load(self, i: int) -> tuple[np.ndarray, int]:
        row = self.df.iloc[i]
        x, sr = read_wav(self.root / row["path"])
        if self.transform is not None:
            x = self.transform(x, sr, i)
        return x, sr

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        for i in range(len(self)):
            yield self.load(i)

    def select(self, mask: np.ndarray | pd.Series) -> SegmentSet:
        return SegmentSet(self.df[np.asarray(mask, dtype=bool)], self.root, self.transform, self.name)

    def with_transform(self, transform: Transform | None, name: str | None = None) -> SegmentSet:
        return SegmentSet(self.df, self.root, transform, self.name if name is None else name)

    def __repr__(self) -> str:
        n1 = int(self.labels.sum()) if len(self) else 0
        return f"SegmentSet({self.name or '?'}: n={len(self)}, faux={n1}, réel={len(self) - n1})"


def make_pair_set(index: pd.DataFrame, root: str | Path, split: str, fake_source: str,
                  real_source: str, name: str | None = None) -> SegmentSet:
    """Réel apparié + faux d'un codec pour un split : les deux branches issues des mêmes pistes."""
    keep = index["split"].eq(split) & index["source"].isin([fake_source, real_source])
    sub = index[keep]
    return SegmentSet(sub, root, None, name or f"{split}:{real_source}|{fake_source}")


def combine(sets: list[SegmentSet], name: str = "") -> SegmentSet:
    """Réunit des SegmentSet de racines différentes (les chemins deviennent absolus)."""
    frames = []
    for s in sets:
        d = s.df.copy()
        d["path"] = [str((s.root / p).resolve()) for p in d["path"]]
        frames.append(d)
    return SegmentSet(pd.concat(frames, ignore_index=True), Path("."), None, name)
