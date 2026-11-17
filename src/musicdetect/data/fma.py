"""Free Music Archive (FMA) : métadonnées, sélection du sous-ensemble, split par artiste.

Inspiré de `mdeff/fma/utils.py` (chargement de tracks.csv à deux niveaux d'en-tête, chemins
`fma_small/000/000002.mp3`), sans ses dépendances (pydot, requests, dotenv) ni ses constructions
incompatibles avec pandas >= 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SUBSET_RANK = {"small": 0, "medium": 1, "large": 2, "full": 3}
_COLUMNS = {
    ("artist", "id"): "artist_id",
    ("set", "split"): "fma_split",
    ("set", "subset"): "subset",
    ("track", "genre_top"): "genre_top",
    ("track", "duration"): "duration_s",
}


def audio_path(audio_dir: str | Path, track_id: int) -> Path:
    tid = f"{int(track_id):06d}"
    return Path(audio_dir) / tid[:3] / f"{tid}.mp3"


def load_tracks(tracks_csv: str | Path) -> pd.DataFrame:
    """Lit `tracks.csv` et ne garde que les colonnes utiles (identifiants, artiste, sous-ensemble, split)."""
    raw = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
    df = pd.DataFrame({new: raw[old] for old, new in _COLUMNS.items()})
    df.index.name = "track_id"
    df = df.reset_index()
    df["track_id"] = df["track_id"].astype(int)
    df["artist_id"] = df["artist_id"].astype(int)
    df["subset"] = df["subset"].astype(str)
    return df


def select_subset(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Les sous-ensembles FMA sont emboîtés : `medium` contient `small`."""
    if subset not in SUBSET_RANK:
        raise ValueError(f"sous-ensemble FMA inconnu : {subset}")
    rank = df["subset"].map(SUBSET_RANK)
    return df[rank <= SUBSET_RANK[subset]].reset_index(drop=True)


def assert_group_disjoint(df: pd.DataFrame, group_col: str = "artist_id", split_col: str = "split") -> None:
    """Lève ValueError si un même groupe (artiste) apparaît dans plusieurs splits."""
    n_splits = df.groupby(group_col)[split_col].nunique()
    bad = n_splits[n_splits > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} groupes apparaissent dans plusieurs splits (fuite), ex. {list(bad.index[:5])}")


def group_split(df: pd.DataFrame, group_col: str, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    """Split train / val / test sans chevauchement de groupes (GroupShuffleSplit en deux temps)."""
    groups = df[group_col].to_numpy()
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    rest, test = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1.0 - test_frac), random_state=seed + 1)
    tr, va = next(gss2.split(rest, groups=groups[rest]))
    out = np.empty(len(df), dtype=object)
    out[rest[tr]], out[rest[va]], out[test] = "train", "val", "test"
    return pd.Series(out, index=df.index, name="split")


def make_manifest(fma_root: str | Path, subset: str = "small", split_mode: str = "artist",
                  val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 0,
                  max_tracks: int | None = None, audio_dir: str | Path | None = None,
                  tracks_csv: str | Path | None = None, check_files: bool = True) -> pd.DataFrame:
    """Manifeste des pistes réelles : track_id, artist_id, genre_top, split, path (MP3 existant)."""
    root = Path(fma_root)
    tracks_csv = Path(tracks_csv) if tracks_csv else root / "fma_metadata" / "tracks.csv"
    audio_dir = Path(audio_dir) if audio_dir else root / f"fma_{subset}"
    df = select_subset(load_tracks(tracks_csv), subset)
    df["path"] = [str(audio_path(audio_dir, t)) for t in df["track_id"]]
    if check_files:
        exists = np.array([Path(p).is_file() for p in df["path"]])
        if not exists.all():
            print(f"[fma] {int((~exists).sum())} fichiers absents sur {len(df)} : ignorés")
        df = df[exists].reset_index(drop=True)
    if max_tracks is not None and max_tracks < len(df):
        df = df.sample(n=int(max_tracks), random_state=seed).sort_values("track_id").reset_index(drop=True)
    if split_mode == "artist":
        df["split"] = group_split(df, "artist_id", val_frac, test_frac, seed)
    elif split_mode == "official":
        df["split"] = df["fma_split"].map({"training": "train", "validation": "val", "test": "test"})
    else:
        raise ValueError(f"split_mode inconnu : {split_mode}")
    assert_group_disjoint(df)       # lève une erreur si le split « officiel » partage des artistes
    return df
