import pytest

from musicdetect.selftest import run_selftest


@pytest.mark.ffmpeg
@pytest.mark.slow
def test_selftest_end_to_end_numpy_only(tmp_path):
    """Faux FMA (MP3) -> chaîne -> codec jouet -> M1 / M0 -> évaluation + attaques, sans PyTorch."""
    out = run_selftest(tmp_path / "st", n_tracks=18, n_clips=12, verbose=False)
    def get(det, test):
        return float(out[(out.detector == det) & (out.test == test)]["auc"].iloc[0])

    assert get("m1", "fma:toy_comb_75hz") > 0.95            # le peigne jouet est détecté
    assert get("m1", "fmc:toygen") > 0.95                    # et il se transfère aux clips 16 kHz
    assert abs(get("m1", "fma:identity_24k") - 0.5) < 1e-9   # contrôle négatif : codec identité
    assert (tmp_path / "st" / "results" / "main" / "attacks_m1__toy_comb_75hz.csv").exists()


@pytest.mark.ffmpeg
@pytest.mark.slow
def test_build_is_resumable_and_adding_a_codec_keeps_the_previous_one(tmp_path):
    import numpy as np

    from musicdetect.config import load_config
    from musicdetect.pipeline import get_paths, step_build_fma, step_manifest
    from musicdetect.selftest import make_fake_fma
    make_fake_fma(tmp_path / "raw", 4, 4, 21.0, 22050, np.random.default_rng(0))
    try:
        import soxr  # noqa: F401
        backend = "soxr"
    except ImportError:
        backend = "scipy"
    cfg = load_config(overrides=[f"paths.fma_root={tmp_path / 'raw'}", f"paths.work_dir={tmp_path / 'w'}",
                                 f"audio.resample_backend={backend}", "fma.val_frac=0.25", "fma.test_frac=0.25"])
    paths = get_paths(cfg)
    step_manifest(cfg, paths)
    a = step_build_fma(cfg, paths, ["toy_comb_75hz"])
    assert set(a["source"]) == {"real_sr24000", "toy_comb_75hz"}
    b = step_build_fma(cfg, paths, ["identity_24k"])
    assert set(b["source"]) == {"real_sr24000", "toy_comb_75hz", "identity_24k"}
    assert not b["seg_id"].duplicated().any()
    n_tracks = b["track_id"].nunique()
    assert (b.groupby("source").size() == b.groupby("source").size().iloc[0]).all() and n_tracks == 4
    c = step_build_fma(cfg, paths, ["identity_24k"])             # reprise : rien à refaire
    assert len(c) == len(b)
