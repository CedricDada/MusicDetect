import numpy as np
import pytest

from musicdetect.audio.attacks import (
    AttackTransform,
    add_noise,
    apply_attack,
    attack_name,
    default_attack_grid,
    mp3_roundtrip,
    pitch_shift,
    resample_roundtrip,
    speed,
)
from musicdetect.config import load_config

SR = 16000


def _tone(f, seconds=4.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _peak_hz(y):
    n = min(len(y), SR * 3)
    spec = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
    return np.fft.rfftfreq(n, 1 / SR)[np.argmax(spec)]


def test_speed_scales_frequency_and_duration(backend):
    y = speed(_tone(440), SR, 1.1, backend)
    assert abs(_peak_hz(y) - 484) < 3
    assert abs(len(y) - len(_tone(440)) / 1.1) < 5


@pytest.mark.ffmpeg
def test_pitch_shift_keeps_duration_and_moves_pitch():
    y = pitch_shift(_tone(440), SR, 2.0)
    assert len(y) == len(_tone(440))
    assert abs(_peak_hz(y) - 440 * 2 ** (2 / 12)) < 6


@pytest.mark.ffmpeg
@pytest.mark.parametrize("kbps", [64, 320])
def test_mp3_roundtrip_is_aligned(kbps, backend, rng):
    t = np.arange(SR * 4) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1234 * t * (1 + 0.1 * np.sin(t)))
         + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    y = mp3_roundtrip(x, SR, kbps, backend=backend)
    assert len(y) == len(x)
    assert np.corrcoef(x, y)[0, 1] > 0.98          # le retard d'encodeur est compensé


def test_resample_roundtrip_removes_high_frequencies(backend):
    x = _tone(7000)
    y = resample_roundtrip(x, SR, 11025, backend)   # 7 kHz > 11025 / 2
    assert len(y) == len(x) and np.sqrt(np.mean(y ** 2)) < 0.05 * np.sqrt(np.mean(x ** 2))


def test_add_noise_snr(rng):
    x = _tone(440)
    y = add_noise(x, 20.0, rng)
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert abs(snr - 20.0) < 0.5


def test_attack_transform_is_deterministic_per_index():
    x = _tone(440)
    t = AttackTransform({"kind": "noise", "snr_db": 10.0}, seed=1, backend="scipy")
    assert np.array_equal(t(x, SR, 5), t(x, SR, 5)) and not np.array_equal(t(x, SR, 5), t(x, SR, 6))


def test_names_and_grid():
    assert attack_name({"kind": "pitch", "semitones": -2}) == "pitch_-2"
    grid = default_attack_grid(dict(load_config().eval.attacks))
    kinds = {g["kind"] for g in grid}
    assert kinds == {"speed", "pitch", "mp3", "resample", "noise"}
    with pytest.raises(ValueError):
        apply_attack(_tone(440), SR, {"kind": "inconnue"})
