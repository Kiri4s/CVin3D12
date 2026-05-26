"""Per-segment geometric analysis via PCA and normal estimation."""
from __future__ import annotations

import numpy as np
import open3d as o3d


def analyse(segment: dict) -> dict:
    xyz = segment["xyz"]
    n = len(xyz)

    cov = np.cov(xyz.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]  # descending: e1 >= e2 >= e3
    e1, e2, e3 = eigvals

    linearity  = (e1 - e2) / e1 if e1 > 0 else 0.0
    planarity  = (e2 - e3) / e1 if e1 > 0 else 0.0
    sphericity = e3 / e1        if e1 > 0 else 0.0

    # Normal consistency via open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    radius = _mean_nn_distance(xyz) * 3
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    # Normal consistency: mean |dot of adjacent normals| — use std of normals as proxy
    normal_std = normals.std(axis=0).mean() if len(normals) > 1 else 1.0

    density = n / (np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)) + 1e-9)

    return {
        "n_points": n,
        "density": float(density),
        "linearity": float(linearity),
        "planarity": float(planarity),
        "sphericity": float(sphericity),
        "normal_std": float(normal_std),
        "mean_nn_dist": float(_mean_nn_distance(xyz)),
    }


def _mean_nn_distance(xyz: np.ndarray) -> float:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    return float(dists.mean()) if len(dists) > 0 else 1.0
