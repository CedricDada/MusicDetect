import numpy as np

from musicdetect.audio.resample import resample
from musicdetect.codecs.toy import ToyCombCodec
from musicdetect.detectors.features import (
    FakeprintConfig,
    effective_bpo,
    fakeprint,
    log_grid,
    lower_hull,
    m0_residual_features,
    power_db_frames,
    to_log_frequency,
)
from musicdetect.detectors.m3_crosscorr import lag_target, xcorr_same_numpy


def _contrast(fp, freqs, f0, n_lo=14, n_hi=105):
    on = np.mean([fp[np.argmin(abs(freqs - f0 * n))] for n in range(n_lo, n_hi)])
    off = np.mean([fp[np.argmin(abs(freqs - f0 * (n + 0.5)))] for n in range(n_lo, n_hi)])
    return on - off


def test_power_db_frames_shape_and_tone_bin():
    sr, n_fft = 16000, 2048
    x = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr).astype(np.float32)
    db = power_db_frames(x, n_fft, 512)
    assert db.shape == (n_fft // 2 + 1, 1 + sr // 512)
    assert abs(np.argmax(db.mean(axis=1)) * sr / n_fft - 1000) < sr / n_fft


def test_lower_hull_touches_endpoints_and_follows_valleys():
    y = np.array([5, 3, 6, 7, 2, 8, 9, 4, 10.0])
    h = lower_hull(y, area=3)
    assert h.shape == y.shape and h[0] == y[0] and h[-1] == y[-1] and h.min() == y.min()
    assert h[4] == 2.0                                  # le creux global est une ancre
    assert np.allclose(lower_hull(np.ones(50), 10), 1.0)
    # un pic étroit sur un plancher plat est entièrement au-dessus de l'enveloppe
    p = np.zeros(200)
    p[100] = 5.0
    assert (p - lower_hull(p, 20)).max() == 5.0


def test_comb_is_visible_in_fakeprint_and_clean_is_not(colored_noise):
    x, sr = colored_noise
    cfg = FakeprintConfig()
    fr, fp_real = fakeprint(resample(x, sr, 16000, "scipy"), 16000, cfg)
    faux = ToyCombCodec(sr, 75.0, -26.0).reconstruct(x)
    _, fp_fake = fakeprint(resample(faux, sr, 16000, "scipy"), 16000, cfg)
    assert _contrast(fp_fake, fr, 75.0) > 0.12
    assert abs(_contrast(fp_real, fr, 75.0)) < 0.06
    assert fp_fake.shape == fp_real.shape == fr.shape and fr[0] >= 1000 and fr[-1] <= 8000


def test_fakeprint_normalisations(colored_noise):
    x, sr = colored_noise
    z = resample(x, sr, 16000, "scipy")
    assert fakeprint(z, 16000, FakeprintConfig(normalise="max"))[1].max() <= 1.0 + 1e-6
    assert abs(np.linalg.norm(fakeprint(z, 16000, FakeprintConfig(normalise="l2"))[1]) - 1) < 1e-4
    assert fakeprint(z, 16000, FakeprintConfig(normalise="none"))[1].min() >= 0


def test_log_grid_odd_and_effective_bpo():
    g = log_grid((1000.0, 8000.0), 1920)
    assert len(g) % 2 == 1 and g[0] == 1000.0 and abs(g[-1] - 8000.0) < 1e-6
    assert abs(effective_bpo(g) - 1920) < 1.0
    assert np.all(np.diff(np.log(g)) > 0)


def test_speed_attack_shifts_log_fakeprint_by_predicted_lag(colored_noise):
    """Prémisse de M3 : x(a·t) décale le fakeprint de log2(a) octaves ; le pic de corrélation le retrouve."""
    from musicdetect.audio.attacks import speed
    x, sr = colored_noise
    comb = ToyCombCodec(sr, 75.0, -18.0).reconstruct(x)
    z = resample(comb, sr, 16000, "scipy")
    cfg = FakeprintConfig(normalise="none")
    grid = log_grid(cfg.band_hz, 1920)
    bpo = effective_bpo(grid)
    f0, fp0 = fakeprint(z, 16000, cfg)
    ref = to_log_frequency(fp0, f0, grid)
    for a in (0.9, 1.1):
        f1, fp1 = fakeprint(speed(z, 16000, a, "scipy"), 16000, cfg)
        cur = to_log_frequency(fp1, f1, grid)
        peak = int(np.argmax(xcorr_same_numpy(cur, ref)))
        assert abs(peak - lag_target(a, len(grid), bpo)) <= 12, (a, peak, lag_target(a, len(grid), bpo))


def test_xcorr_same_numpy_peak_convention():
    rng = np.random.default_rng(0)
    F = 1001
    w = rng.standard_normal(F)
    for s in (-40, 0, 25):
        assert int(np.argmax(xcorr_same_numpy(np.roll(w, s), w))) == F // 2 + s


def test_m0_residual_features_are_finite_and_discriminate_silence():
    sr = 16000
    x = 0.1 * np.random.default_rng(0).standard_normal(sr * 4).astype(np.float32)
    f1 = m0_residual_features(x, sr)
    y = x.copy()
    y[: sr] = 0
    f2 = m0_residual_features(y, sr)
    assert np.all(np.isfinite(f1)) and f2[5] > f1[5]          # ratio de silence de début
