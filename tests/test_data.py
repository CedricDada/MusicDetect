import numpy as np
import pandas as pd
import pytest

from musicdetect.data import fma, fmc
from musicdetect.data.segments import SegmentSet, combine, make_pair_set, read_index, write_index

TRACKS = """,artist,set,set,track,track
,id,split,subset,genre_top,duration
track_id,,,,,
2,1,training,small,Hip-Hop,30
3,1,training,small,Hip-Hop,30
5,2,validation,medium,Rock,30
10,3,test,large,Pop,30
"""


def test_audio_path_matches_fma_convention():
    assert str(fma.audio_path("data/fma_small", 2)) == "data/fma_small/000/000002.mp3"
    assert str(fma.audio_path("d", 133297)) == "d/133/133297.mp3"


def test_load_tracks_and_subset_nesting(tmp_path):
    p = tmp_path / "tracks.csv"
    p.write_text(TRACKS)
    df = fma.load_tracks(p)
    assert list(df["track_id"]) == [2, 3, 5, 10] and list(df["artist_id"]) == [1, 1, 2, 3]
    assert len(fma.select_subset(df, "small")) == 2
    assert len(fma.select_subset(df, "medium")) == 3 and len(fma.select_subset(df, "large")) == 4
    with pytest.raises(ValueError):
        fma.select_subset(df, "xxl")


def test_group_split_has_no_artist_overlap_and_is_reproducible():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"artist_id": rng.integers(0, 50, 400)})
    s1 = fma.group_split(df, "artist_id", 0.1, 0.2, seed=3)
    s2 = fma.group_split(df, "artist_id", 0.1, 0.2, seed=3)
    assert (s1 == s2).all()
    df["split"] = s1
    fma.assert_group_disjoint(df)
    assert set(s1) == {"train", "val", "test"}
    assert 0.1 < (s1 == "test").mean() < 0.35


def test_assert_group_disjoint_raises_on_leak():
    df = pd.DataFrame({"artist_id": [1, 1, 2], "split": ["train", "test", "train"]})
    with pytest.raises(ValueError):
        fma.assert_group_disjoint(df)


def test_fmc_discover_maps_directories_to_generators(tmp_path):
    for d, n in [("MusicCaps", 2), ("MusicGen_medium", 2), ("other", 1)]:
        (tmp_path / "x" / d).mkdir(parents=True)
        for k in range(n):
            (tmp_path / "x" / d / f"id{k}.wav").write_bytes(b"")
    df = fmc.discover(tmp_path, {"real": ["MusicCaps"], "musicgen": ["MusicGen_medium"]})
    assert set(df["generator"]) == {"real", "musicgen"} and len(df) == 4
    assert df.loc[df.generator == "real", "label"].eq(0).all() and df.loc[df.generator == "musicgen", "label"].eq(1).all()
    assert "ATTENTION" not in fmc.summary(df)
    assert "ATTENTION" in fmc.summary(df[df.generator != "real"])


def test_segment_index_roundtrip_and_sets(tmp_path):
    df = pd.DataFrame({"seg_id": ["a", "b", "c"], "path": ["a.wav", "b.wav", "c.wav"], "label": [0, 1, 1],
                       "source": ["real_sr24000", "toy", "toy"], "group": ["1", "1", "2"],
                       "split": ["train", "train", "val"], "pair_id": ["p", "p", "q"]})
    write_index(df, tmp_path / "i.csv")
    back = read_index(tmp_path / "i.csv")
    assert list(back["seg_id"]) == ["a", "b", "c"] and "meta_codec" in back.columns
    s = make_pair_set(back, tmp_path, "train", "toy", "real_sr24000")
    assert len(s) == 2 and list(s.labels) == [0, 1] and s.groups.tolist() == ["1", "1"]
    c = combine([s, SegmentSet(back[back.split == "val"], tmp_path)])
    assert len(c) == 3 and all(p.startswith("/") for p in c.df["path"])
