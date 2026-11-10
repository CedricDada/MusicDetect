"""Chargement de la configuration (YAML par défaut + fichier utilisateur + surcharges CLI)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


class AttrDict(dict):
    """dict à accès par attribut (lecture et écriture), récursif sur les sous-dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - message d'erreur seulement
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def deep_merge(base: dict, extra: dict) -> dict:
    """Fusion récursive : les clés de `extra` écrasent celles de `base`."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(cfg: dict, dotted_key: str, raw_value: str) -> None:
    """Applique `a.b.c=valeur` ; la valeur est interprétée comme du YAML (nombre, liste, null...)."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"{dotted_key}: {k} n'est pas une section")
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> AttrDict:
    """Charge `default.yaml`, fusionne `path` s'il est donné, puis applique `overrides` (`k.sub=val`)."""
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"surcharge invalide (attendu cle=valeur) : {item!r}")
        key, value = item.split("=", 1)
        apply_override(cfg, key.strip(), value)
    return _wrap(cfg)
