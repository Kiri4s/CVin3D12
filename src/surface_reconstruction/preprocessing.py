"""Noise removal, normalisation, optional voxel downsampling."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from omegaconf import DictConfig


def preprocess(xyz: np.ndarray, labels: np.ndarray, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if cfg.statistical_outlier.enabled:
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=cfg.statistical_outlier.nb_neighbors,
            std_ratio=cfg.statistical_outlier.std_ratio,
        )
        pcd = pcd.select_by_index(ind)
        labels = labels[ind]

    xyz_clean = np.asarray(pcd.points)

    if cfg.normalize.enabled:
        centre = xyz_clean.mean(axis=0)
        scale = xyz_clean.std()
        xyz_clean = (xyz_clean - centre) / (scale if scale > 0 else 1.0)
        pcd.points = o3d.utility.Vector3dVector(xyz_clean)

    if cfg.voxel_downsample.enabled:
        # Keep label per voxel by tracking index of nearest original point
        pcd_down, _, idx_map = pcd.voxel_down_sample_and_trace(
            cfg.voxel_downsample.voxel_size,
            pcd.get_min_bound(),
            pcd.get_max_bound(),
        )
        voxel_labels = np.array(
            [labels[list(indices)[0]] for indices in idx_map]
        )
        return np.asarray(pcd_down.points), voxel_labels

    return np.asarray(pcd.points), labels
