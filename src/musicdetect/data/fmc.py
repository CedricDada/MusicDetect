"""FakeMusicCaps (Comanducci et al.) : découverte des fichiers et mise en index.

La disposition exacte de l'archive Zenodo n'a pas été vérifiée lors de l'écriture : on cherche tous les
WAV sous `fmc_root` et on les rattache à un générateur d'après le nom d'un dossier parent (alias dans
`fmc.generators`). `musicdetect fmc-index` imprime ce qui a été trouvé, à contrôler en semaine 1.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def discover(root: str | Path, generators: dict[str, list[str]], group_regex: str | None = None) -> pd.DataFrame:
    """Colonnes : generator, label (0 = clip réel MusicCaps, 1 = généré), path (relatif à root), clip_id, group."""
    root = Path(root)
    alias = {a.lower(): canon for canon, names in generators.items() for a in names}
    rx = re.compile(group_regex) if group_regex else None
    rows = []
    for p in sorted(root.rglob("*.wav")):
        rel = p.relative_to(root)
        canon = next((alias[part.lower()] for part in reversed(rel.parts[:-1]) if part.lower() in alias), None)
        if canon is None:
            continue
        stem = p.stem
        gid = rx.search(stem).group(1) if rx and rx.search(stem) else stem
        rows.append({"generator": canon, "label": 0 if canon == "real" else 1,
                     "path": str(rel), "clip_id": stem, "group": gid})
    return pd.DataFrame(rows, columns=["generator", "label", "path", "clip_id", "group"])


def summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "[fmc] aucun WAV rattaché à un générateur : vérifier fmc_root et fmc.generators"
    lines = [f"[fmc] {len(df)} clips"] + [f"  {g:<20s} {n:6d}" for g, n in df.groupby("generator").size().items()]
    if "real" not in set(df["generator"]):
        lines.append("  ATTENTION : pas de clips réels (MusicCaps) dans l'archive ; fournir des clips réels "
                     "à part ou utiliser FMA test comme réel (diagnostiquer la fuite de format avec M0).")
    return "\n".join(lines)
