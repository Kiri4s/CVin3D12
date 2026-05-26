"""Map geometry type to reconstruction method name."""
from __future__ import annotations

from omegaconf import DictConfig


def select_method(geo_type: str, cfg: DictConfig) -> str:
    return cfg.method_map[geo_type]
