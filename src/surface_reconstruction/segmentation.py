"""Group points by label; drop segments below min_points."""
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def segment(xyz: np.ndarray, labels: np.ndarray, cfg: DictConfig) -> list[dict]:
    segments = []
    for label in np.unique(labels):
        mask = labels == label
        pts = xyz[mask]
        if len(pts) >= cfg.min_points:
            segments.append({"label": int(label), "xyz": pts})
    return segments
