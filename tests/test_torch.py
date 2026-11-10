"""Tests des parties PyTorch (M2, M3, modules). Ignorés sans l'extra « torch » ; lancer : pytest -m torch."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from musicdetect.audio.io import write_wav  # noqa: E402
from musicdetect.codecs.toy import ToyCombCodec  # noqa: E402
from musicdetect.config import load_config  # noqa: E402
from musicdetect.data.segments import SegmentSet  # noqa: E402
from musicdetect.detectors import nn_models  # noqa: E402
from musicdetect.detectors.features import FakeprintConfig  # noqa: E402
from musicdetect.detectors.m2_cnn import SpecCNNDetector  # noqa: E402
from musicdetect.detectors.m3_crosscorr import CrossCorrDetector, xcorr_same_numpy  # noqa: E402

pytestmark = pytest.mark.torch


def _tiny_set(tmp_path, n=8, seconds=3.0):
    rng = np.random.default_rng(0)
    sr, nn = 16000, int(seconds * 16000)
    toy = ToyCombCodec(sr, 75.0, -18.0)
    rows = []
    for k in range(n):
        spec = np.fft.rfft(rng.standard_normal(nn))
        f = np.fft.rfftfreq(nn, 1 / sr)
        spec[1:] /= np.sqrt(f[1:] / 100.0)
        x = np.fft.irfft(spec, nn).astype(np.float32)
        x /= np.abs(x).max() * 2
        for label, sig in ((0, x), (1, toy.reconstruct(x))):
            p = f"{label}_{k}.wav"
            write_wav(tmp_path / p, sig, sr)
            rows.append({"seg_id": p, "path": p, "label": label, "group": str(k)})
    return SegmentSet(pd.DataFrame(rows), tmp_path, None, "tiny")


def test_spec_cnn_shapes_and_variable_length():
    m = nn_models.SpecCNN(n_freq=65, hidden=[8, 8], strides=[1, 3], kernel=3, head_dense=4).eval()
    assert m(torch.randn(2, 65, 100)).shape == (2,)
    assert m(torch.randn(1, 65, 57)).shape == (1,)


def test_xcorr_probe_matches_numpy_reference():
    torch.manual_seed(0)
    F = 101
    p = nn_models.XCorrProbe(F, use_norm=False, use_bias=False, max_clamp=1e9)
    x = torch.randn(3, F)
    logits, cc = p(x, convolve=True)
    w = p.weights.detach().numpy().ravel()
    ref = np.stack([xcorr_same_numpy(xi.numpy(), w) for xi in x])
    np.testing.assert_allclose(cc.detach().numpy(), ref, atol=1e-4)
    np.testing.assert_allclose(logits.detach().numpy(), ref.max(axis=1), atol=1e-4)


def test_xcorr_probe_peak_follows_shift_and_even_size_is_rejected():
    F = 201
    p = nn_models.XCorrProbe(F, use_norm=False, use_bias=False, max_clamp=1e9)
    w = p.weights.detach().clone() * 50
    p.weights.data = w
    for s in (-30, 0, 17):
        x = torch.roll(w, s, dims=1)
        _, cc = p(x, convolve=True)
        assert int(cc.argmax(dim=1)) == F // 2 + s
    with pytest.raises(ValueError):
        nn_models.XCorrProbe(200)


def test_m3_fit_score_save_load_roundtrip(tmp_path):
    segs = _tiny_set(tmp_path)
    cfg = dict(load_config().m3)
    cfg.update({"bins_per_octave": 120, "epochs": 2, "aug_per_segment": 1, "batch_size": 4, "patience": 5})
    det = CrossCorrDetector(FakeprintConfig(), cfg, backend="scipy", device="cpu")
    det.fit(segs, segs)
    s = det.score(segs)
    assert s.shape == (len(segs),) and np.all((s >= 0) & (s <= 1))
    det.save(tmp_path / "m3.pt")
    again = CrossCorrDetector.load(tmp_path / "m3.pt", "cpu")
    np.testing.assert_allclose(again.score(segs), s, atol=1e-5)


def test_m2_fit_score_save_load_roundtrip(tmp_path):
    segs = _tiny_set(tmp_path)
    cfg = dict(load_config().m2)
    cfg.update({"n_fft": 256, "hop": 128, "hidden": [4, 4], "strides": [1, 3], "head_dense": 4, "epochs": 1,
                "batch_size": 4, "crop_s": 1.0})
    det = SpecCNNDetector(cfg, device="cpu")
    det.fit(segs, segs)
    s = det.score(segs)
    assert s.shape == (len(segs),) and np.all((s >= 0) & (s <= 1))
    det.save(tmp_path / "m2.pt")
    again = SpecCNNDetector.load(tmp_path / "m2.pt", "cpu")
    np.testing.assert_allclose(again.score(segs), s, atol=1e-5)
