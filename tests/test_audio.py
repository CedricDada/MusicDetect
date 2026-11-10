import numpy as np

from musicdetect.audio import (
    ChainConfig,
    estimate_delay,
    fit_length,
    process_external_clip,
    process_fma_track,
    read_wav,
    resample,
    write_wav,
)
from musicdetect.audio.chain import normalise_level, segment_starts, silence_bounds
from musicdetect.codecs.toy import IdentityCodec, ToyCombCodec


def test_wav_roundtrip_int16(tmp_path, rng):
    x = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    write_wav(tmp_path / "a.wav", x, 16000)
    y, sr = read_wav(tmp_path / "a.wav")
    assert sr == 16000 and len(y) == len(x)
    assert np.abs(y - x).max() < 1.0 / 32768 + 1e-6


def test_wav_float32_exact(tmp_path, rng):
    x = rng.uniform(-1, 1, 1000).astype(np.float32)
    write_wav(tmp_path / "a.wav", x, 8000, dtype="float32")
    assert np.array_equal(read_wav(tmp_path / "a.wav")[0], x)


def test_resample_preserves_tone_frequency(backend):
    sr = 24000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = resample(x, sr, 16000, backend)
    assert abs(len(y) - 16000) <= 2
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(y)))] - 1000) < 2


def test_resample_same_rate_is_copy():
    x = np.arange(10, dtype=np.float32)
    y = resample(x, 16000, 16000)
    assert np.array_equal(x, y) and y is not x


def test_fit_length():
    assert len(fit_length(np.ones(5), 8)) == 8 and len(fit_length(np.ones(9), 8)) == 8


def test_estimate_delay_both_signs(rng):
    x = rng.standard_normal(16000).astype(np.float32)
    assert estimate_delay(x, np.pad(x, (37, 0))[:16000], 800) == 37
    assert estimate_delay(x, x[25:], 800) == -25


def test_normalise_level_hits_target_and_limits():
    x = (np.sin(np.linspace(0, 200, 16000)) * 0.01).astype(np.float32)
    y, g, limited = normalise_level(x, -23.0, 0.99)
    assert not limited
    assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) + 23.0) < 0.01
    z = np.zeros(16000, dtype=np.float32)
    z[100] = 0.5                                   # pic très haut devant un RMS minuscule
    z[:] += 1e-3 * np.sin(np.arange(16000))
    _, _, lim2 = normalise_level(z, -10.0, 0.99)
    assert lim2


def test_silence_bounds():
    sr = 16000
    x = np.zeros(sr * 3, dtype=np.float32)
    x[sr:2 * sr] = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    a, b = silence_bounds(x, sr, -50.0)
    assert abs(a - sr) < 0.05 * sr and abs(b - 2 * sr) < 0.05 * sr


def test_segment_starts_contiguous_with_jitter():
    rng = np.random.default_rng(1)
    s = segment_starts(100, 30, 3, rng)
    assert len(s) == 3 and s[1] - s[0] == 30 and s[-1] + 30 <= 100
    assert segment_starts(20, 30, 3, rng) == []


def test_process_fma_track_pairs_are_aligned_and_labelled(colored_noise, backend):
    x, sr = colored_noise                                  # 24 kHz
    cfg = ChainConfig(segments_per_track=1, backend=backend, silence_policy="none")
    codecs = {"toy": ToyCombCodec(24000, 75.0, -26.0), "ident": IdentityCodec(24000)}
    segs = process_fma_track(x[:sr * 12], sr, codecs, cfg, track_id=7)
    by = {s.source: s for s in segs}
    assert set(by) == {"real_sr24000", "toy", "ident"}
    assert by["toy"].label == 1 and by["real_sr24000"].label == 0
    L = cfg.segment_len
    assert all(len(s.audio) == L for s in segs)
    assert by["toy"].start_s == by["real_sr24000"].start_s
    # le faux « identité » est exactement le réel après la chaîne (même aller-retour, même niveau)
    assert np.max(np.abs(by["ident"].audio - by["real_sr24000"].audio)) < 1e-5
    # le niveau est normalisé des deux côtés
    for s in segs:
        assert abs(20 * np.log10(np.sqrt(np.mean(s.audio ** 2))) - cfg.level_dbfs) < 0.5 or s.limited


def test_process_fma_track_is_deterministic(colored_noise, backend):
    x, sr = colored_noise
    cfg = ChainConfig(segments_per_track=1, backend=backend)
    c = {"t": ToyCombCodec()}
    a = process_fma_track(x[:sr * 12], sr, c, cfg, 3)
    b = process_fma_track(x[:sr * 12], sr, c, cfg, 3)
    assert all(np.array_equal(p.audio, q.audio) and p.start_s == q.start_s for p, q in zip(a, b, strict=True))


def test_process_external_clip_short_clip_kept(backend):
    cfg = ChainConfig(backend=backend)
    x = (np.random.default_rng(0).standard_normal(16000 * 4) * 0.1).astype(np.float32)
    out = process_external_clip(x, 16000, cfg, "fmc:g", 1)
    assert len(out) == 1 and out[0].extra.get("short") and out[0].label == 1


def test_ablation_without_chain_keeps_raw_level(backend):
    cfg = ChainConfig(backend=backend, normalise_level=False)
    x = (np.random.default_rng(0).standard_normal(16000 * 10) * 0.05).astype(np.float32)
    out = process_external_clip(x, 16000, cfg, "fmc:g", 1)
    assert np.allclose(out[0].audio, x[: cfg.segment_len], atol=1e-6)
