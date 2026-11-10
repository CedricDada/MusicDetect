"""Fabrique de codecs à partir de la section `codecs` de la configuration."""

from __future__ import annotations

from .base import Codec
from .toy import IdentityCodec, ToyCombCodec


def build_codec(name: str, spec: dict, device: str | None = None) -> Codec:
    """Instancie un codec d'après son `kind`. Les codecs neuronaux chargent leurs poids à la 1re utilisation."""
    kind = spec["kind"]
    if kind == "identity":
        return IdentityCodec(native_sr=spec.get("native_sr", 24000), name=name)
    if kind == "toy_comb":
        return ToyCombCodec(native_sr=spec.get("native_sr", 24000), f0=spec.get("f0", 75.0),
                            rel_db=spec.get("rel_db", -26.0), seed=spec.get("seed", 0), name=name)
    from . import neural
    if kind == "encodec":
        return neural.EncodecCodec(spec["hf_id"], spec.get("bandwidth", 6.0), name, device)
    if kind == "musicgen_encodec":
        return neural.MusicgenEncodecCodec(spec["hf_id"], name, device)
    if kind == "dac":
        return neural.DacCodec(spec["hf_id"], spec.get("n_quantizers"), name, device)
    if kind == "mimi":
        return neural.MimiCodec(spec["hf_id"], spec.get("num_quantizers", 8), name, device)
    raise ValueError(f"type de codec inconnu : {kind} (codec {name})")


def build_codecs(cfg_codecs: dict, names: list[str], device: str | None = None) -> dict[str, Codec]:
    missing = [n for n in names if n not in cfg_codecs]
    if missing:
        raise KeyError(f"codecs absents de la configuration : {missing} ; disponibles : {sorted(cfg_codecs)}")
    return {n: build_codec(n, dict(cfg_codecs[n]), device) for n in names}


__all__ = ["Codec", "IdentityCodec", "ToyCombCodec", "build_codec", "build_codecs"]
