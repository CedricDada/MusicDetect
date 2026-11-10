import numpy as np
import pytest

from musicdetect.codecs import build_codec, build_codecs
from musicdetect.codecs.toy import IdentityCodec, ToyCombCodec
from musicdetect.config import load_config


def test_identity_is_exact_copy(rng):
    x = rng.standard_normal(1000).astype(np.float32)
    y = IdentityCodec().reconstruct(x)
    assert np.array_equal(x, y) and y is not x


def test_toy_comb_power_and_period(colored_noise):
    x, sr = colored_noise
    c = ToyCombCodec(sr, 75.0, -26.0)
    d = c.reconstruct(x) - x
    ratio_db = 10 * np.log10(np.mean(d ** 2) / np.mean(x ** 2))
    assert abs(ratio_db + 26.0) < 1.0
    p = sr // 75                                     # 320 échantillons : période exacte
    assert np.allclose(d[:p], d[p:2 * p], atol=1e-5)
    assert c.reconstruct(x).shape == x.shape


def test_toy_comb_non_integer_period_branch(rng):
    x = (rng.standard_normal(8000) * 0.1).astype(np.float32)
    y = ToyCombCodec(16000, 77.7, -20.0).reconstruct(x)
    assert y.shape == x.shape and np.all(np.isfinite(y))


def test_factory_errors_are_explicit():
    cfg = load_config()
    assert build_codec("t", dict(cfg.codecs.toy_comb_75hz)).native_sr == 24000
    with pytest.raises(KeyError):
        build_codecs(dict(cfg.codecs), ["absent"])
    with pytest.raises(ValueError):
        build_codec("x", {"kind": "inconnu"})


def test_neural_codecs_fail_cleanly_without_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        c = build_codec("e", {"kind": "encodec", "hf_id": "facebook/encodec_24khz", "bandwidth": 6.0})
        with pytest.raises(ImportError, match="PyTorch"):
            c.reconstruct(np.zeros(1000, dtype=np.float32))
    else:
        pytest.skip("torch présent : voir tests/test_torch.py")
