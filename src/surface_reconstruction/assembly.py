"""Merge per-segment meshes into one unified model."""
from __future__ import annotations

import open3d as o3d


def assemble(meshes: list[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    combined = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        if len(mesh.vertices) == 0:
            continue
        mesh.compute_vertex_normals()
        combined += mesh
    combined.merge_close_vertices(eps=1e-6)
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.remove_degenerate_triangles()
    if not combined.has_vertex_normals():
        combined.compute_vertex_normals()
    return combined
