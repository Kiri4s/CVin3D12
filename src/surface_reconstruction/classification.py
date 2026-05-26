"""Rule-based geometry type classification from PCA features."""
from __future__ import annotations

from omegaconf import DictConfig


def classify(geo: dict, cfg: DictConfig) -> str:
    """Return one of: planar / tubular / spherical / complex."""
    if geo["planarity"] >= cfg.planarity_threshold:
        return "planar"
    if geo["linearity"] >= cfg.linearity_threshold:
        return "tubular"
    if geo["sphericity"] >= cfg.sphericity_threshold:
        return "spherical"
    return "complex"
