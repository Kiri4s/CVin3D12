"""Surface reconstruction algorithms (Poisson / Alpha Shapes / Ball Pivoting)."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from omegaconf import DictConfig


def reconstruct(segment: dict, geo: dict, method: str, cfg: DictConfig) -> o3d.geometry.TriangleMesh:
    pcd = _make_pcd(segment["xyz"], geo)

    if method == "poisson":
        return _poisson(pcd, cfg.poisson)
    if method == "alpha_shapes":
        return _alpha_shapes(pcd, geo, cfg.alpha_shapes)
    if method == "ball_pivoting":
        return _ball_pivoting(pcd, geo, cfg.ball_pivoting)
    raise ValueError(f"Unknown method: {method}")


def _make_pcd(xyz: np.ndarray, geo: dict) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    radius = geo["mean_nn_dist"] * 3
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def _poisson(pcd: o3d.geometry.PointCloud, cfg: DictConfig) -> o3d.geometry.TriangleMesh:
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=cfg.depth,
        width=cfg.width,
        scale=cfg.scale,
        linear_fit=cfg.linear_fit,
    )
    return mesh


def _alpha_shapes(pcd: o3d.geometry.PointCloud, geo: dict, cfg: DictConfig) -> o3d.geometry.TriangleMesh:
    alpha = cfg.alpha if cfg.alpha > 0 else geo["mean_nn_dist"] * 4
    return o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)


def _ball_pivoting(pcd: o3d.geometry.PointCloud, geo: dict, cfg: DictConfig) -> o3d.geometry.TriangleMesh:
    r = geo["mean_nn_dist"]
    radii = o3d.utility.DoubleVector([r * m for m in cfg.radii_multipliers])
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
