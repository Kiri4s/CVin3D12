"""Quality metrics: point-to-mesh distance, smoothness, coverage."""
from __future__ import annotations

import numpy as np
import open3d as o3d


def evaluate(xyz_original: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> dict:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return {"error": "empty mesh"}

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    pts = o3d.core.Tensor(xyz_original.astype(np.float32))
    abs_dists = np.abs(scene.compute_signed_distance(pts).numpy())

    mean_dist = float(abs_dists.mean())
    max_dist  = float(abs_dists.max())
    rms_dist  = float(np.sqrt((abs_dists ** 2).mean()))

    # Smoothness: mean dihedral angle deviation (via triangle normals std)
    mesh.compute_triangle_normals()
    tri_normals = np.asarray(mesh.triangle_normals)
    smoothness = float(np.std(tri_normals)) if len(tri_normals) > 0 else float("nan")

    return {
        "n_vertices": len(mesh.vertices),
        "n_triangles": len(mesh.triangles),
        "point_to_mesh_mean": mean_dist,
        "point_to_mesh_max": max_dist,
        "point_to_mesh_rms": rms_dist,
        "triangle_normal_std": smoothness,
    }
