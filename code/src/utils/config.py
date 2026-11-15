from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def section(self, key: str) -> "Config":
        return Config(raw=self.raw.get(key, {}))

    def merge(self, other: dict[str, Any]) -> "Config":
        merged = copy.deepcopy(self.raw)
        _deep_update(merged, other)
        return Config(raw=merged)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)


def _deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw)
