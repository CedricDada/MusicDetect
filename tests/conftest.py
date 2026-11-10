import shutil

import numpy as np
import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
try:
    import soxr  # noqa: F401
    BACKEND = "soxr"
except ImportError:
    BACKEND = "scipy"          # repli explicite : les tests ne valident alors pas soxr


def pytest_configure(config):
    config.addinivalue_line("markers", "ffmpeg: nécessite ffmpeg")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "ffmpeg" in item.keywords and not HAS_FFMPEG:
            item.add_marker(pytest.mark.skip(reason="ffmpeg/ffprobe absents"))


@pytest.fixture
def backend():
    return BACKEND


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def colored_noise(rng):
    """10 s de bruit 1/f à 24 kHz : un « fond musical » sans raies (pour tester les peignes)."""
    sr, n = 24000, 240000
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[1:] /= np.sqrt(f[1:] / 100.0)
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (np.abs(x).max() * 2), sr
