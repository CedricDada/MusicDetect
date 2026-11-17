"""Construction des jeux de segments WAV (réel apparié + faux par codec ; clips FakeMusicCaps)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..audio.chain import (
    ChainConfig,
    SegmentRecord,
    process_external_clip,
    process_fma_track,
    real_source_name,
)
from ..audio.io import decode_mono, probe, read_wav, write_wav
from ..utils import tqdm
from .segments import read_index, write_index


def _meta(info: dict) -> dict:
    return {
        "meta_sample_rate": info["sample_rate"], "meta_channels": info["channels"],
        "meta_bit_rate": info["bit_rate"], "meta_duration": info["duration"], "meta_codec": info["codec"],
    }


def _rows(segs: list[SegmentRecord], base: dict, key: str, anchor: int, out_dir: Path, wav_dtype: str) -> list[dict]:
    rows = []
    for s in segs:
        folder = s.source.replace(":", "_")
        rel = f"{folder}/{key}_{s.k}.wav"
        write_wav(out_dir / rel, s.audio, anchor, dtype=wav_dtype)
        rows.append({
            **base, "seg_id": f"{folder}/{key}_{s.k}", "path": rel, "label": s.label, "source": s.source,
            "pair_id": f"{key}_{s.k}", "start_s": s.start_s, "sr": anchor, "gain_db": s.gain_db,
            "limited": int(s.limited), "delay_samples": s.delay_samples,
            "lead_silence_s": s.lead_silence_s, "trail_silence_s": s.trail_silence_s,
        })
    return rows


def build_fma_segments(manifest: pd.DataFrame, codecs: dict, chain: ChainConfig, out_dir: str | Path,
                       index_path: str | Path, max_track_s: float = 31.0, wav_dtype: str = "int16",
                       resume: bool = True, flush_every: int = 100, progress: bool = True) -> pd.DataFrame:
    """Applique la chaîne à chaque piste du manifeste ; écrit les WAV et l'index CSV (reprise possible).

    Les codecs sont préparés d'abord (chargement des poids, fréquence native exacte).
    Les pistes illisibles sont consignées dans `<index>.failures.csv` et ignorées.
    """
    out_dir, index_path = Path(out_dir), Path(index_path)
    for c in codecs.values():
        c.prepare()
    wanted = {real_source_name(int(c.native_sr)) for c in codecs.values()} | set(codecs)

    done_rows: list[dict] = []
    done_tracks: set[int] = set()
    if resume and index_path.exists():
        old = read_index(index_path)
        done_rows = old.to_dict("records")
        have = old.groupby("track_id")["source"].agg(set)
        done_tracks = {int(t) for t, srcs in have.items() if wanted <= srcs}

    # Une piste déjà complète est conservée telle quelle ; pour les autres, on ne retire que les lignes des
    # sources qu'on va régénérer (ajouter un codec plus tard ne perd pas les codecs précédents).
    rows = [r for r in done_rows if int(r["track_id"]) in done_tracks or r["source"] not in wanted]
    failures: list[dict] = []
    it = manifest.itertuples(index=False)
    if progress:
        it = tqdm(it, total=len(manifest), desc="pistes FMA")
    for n, rec in enumerate(it, 1):
        tid = int(rec.track_id)
        if tid in done_tracks:
            continue
        try:
            info = probe(rec.path)
            x, sr = decode_mono(rec.path, max_seconds=max_track_s)
            segs = process_fma_track(x, sr, codecs, chain, tid)
        except Exception as exc:  # piste corrompue, échec codec, etc.
            failures.append({"track_id": tid, "path": rec.path, "error": repr(exc)[:300]})
            continue
        base = {"track_id": tid, "group": str(rec.artist_id), "split": rec.split, **_meta(info)}
        rows += _rows(segs, base, f"{tid:06d}", chain.anchor_sr, out_dir, wav_dtype)
        if n % flush_every == 0:
            write_index(pd.DataFrame(rows), index_path)
    index = pd.DataFrame(rows)
    write_index(index, index_path)
    if failures:
        pd.DataFrame(failures).to_csv(index_path.with_suffix(".failures.csv"), index=False)
        print(f"[build] {len(failures)} pistes en échec -> {index_path.with_suffix('.failures.csv')}")
    return index


def build_fmc_segments(fmc_df: pd.DataFrame, fmc_root: str | Path, chain: ChainConfig, out_dir: str | Path,
                       index_path: str | Path, wav_dtype: str = "int16", progress: bool = True,
                       ) -> pd.DataFrame:
    """Applique les étapes 4 à 6 aux clips FakeMusicCaps (ou, si `chain.normalise_level` est faux, les
    ré-écrit sans normalisation de niveau : « version sans chaîne » de l'ablation H3)."""
    out_dir, index_path, fmc_root = Path(out_dir), Path(index_path), Path(fmc_root)
    rows = []
    it = fmc_df.itertuples(index=False)
    if progress:
        it = tqdm(it, total=len(fmc_df), desc="clips FakeMusicCaps")
    for n, rec in enumerate(it):
        src = fmc_root / rec.path
        try:
            info = probe(src)
            x, sr = read_wav(src)
            segs = process_external_clip(x, sr, chain, f"fmc:{rec.generator}", int(rec.label), clip_key=n)
        except Exception as exc:
            print(f"[fmc] {rec.path} ignoré : {exc!r}")
            continue
        base = {"track_id": n, "group": rec.group, "split": "test", **_meta(info)}
        rows += _rows(segs, base, f"{rec.generator}_{rec.clip_id}", chain.anchor_sr, out_dir, wav_dtype)
    index = pd.DataFrame(rows)
    write_index(index, index_path)
    return index
