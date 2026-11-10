import pytest

from musicdetect.config import deep_merge, load_config


def test_defaults_and_attribute_access():
    cfg = load_config()
    assert cfg.audio.anchor_sr == 16000
    assert cfg.codecs.dac_44k["kind"] == "dac"
    assert cfg.m3.lamb == 0.05


def test_overrides_are_yaml_typed():
    cfg = load_config(overrides=["audio.anchor_sr=8000", "m3.speed_range=[0.5,2.0]", "fma.max_tracks=null"])
    assert cfg.audio.anchor_sr == 8000
    assert cfg.m3.speed_range == [0.5, 2.0]
    assert cfg.fma.max_tracks is None


def test_bad_override_raises():
    with pytest.raises(ValueError):
        load_config(overrides=["pas_de_egal"])


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": 2}}
    b = {"x": {"y": 3}}
    assert deep_merge(a, b) == {"x": {"y": 3, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}
