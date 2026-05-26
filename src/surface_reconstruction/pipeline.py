"""Orchestrator: runs the full reconstruction pipeline for one point cloud."""
from __future__ import annotations

import logging
from pathlib import Path

import open3d as o3d
from omegaconf import DictConfig

import assembly
import classification
import evaluation
import geometry
import method_selector
import preprocessing
import reconstruction
import segmentation
import visualization

log = logging.getLogger(__name__)


def run_cloud(cloud: dict, cfg: DictConfig, out_dir: Path) -> dict:
    name = cloud["name"]
    log.info("[%s] start (%d points)", name, len(cloud["xyz"]))

    xyz, labels = preprocessing.preprocess(cloud["xyz"], cloud["labels"], cfg.preprocessing)
    log.info("[%s] after preprocessing: %d points", name, len(xyz))

    segments = segmentation.segment(xyz, labels, cfg.segmentation)
    log.info("[%s] %d segments after min_points filter", name, len(segments))

    meshes: list[o3d.geometry.TriangleMesh] = []
    mesh_records: list[dict] = []
    seg_metrics: list[dict] = []

    for seg in segments:
        label = seg["label"]
        geo = geometry.analyse(seg)
        geo_type = classification.classify(geo, cfg.classification)
        method = method_selector.select_method(geo_type, cfg.reconstruction)
        log.info("[%s] label=%d  type=%s  method=%s  n=%d", name, label, geo_type, method, geo["n_points"])

        try:
            mesh = reconstruction.reconstruct(seg, geo, method, cfg.reconstruction)
        except Exception as exc:
            log.warning("[%s] label=%d reconstruction failed: %s", name, label, exc)
            continue

        seg_out = out_dir / name / f"segment_{label}.ply"
        seg_out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(seg_out), mesh)

        metrics = evaluation.evaluate(seg["xyz"], mesh)
        metrics.update({"label": label, "geo_type": geo_type, "method": method})
        seg_metrics.append(metrics)
        meshes.append(mesh)
        mesh_records.append({"label": label, "method": method, "mesh": mesh})

    combined = assembly.assemble(meshes)
    combined_path = out_dir / f"{name}_combined.ply"
    o3d.io.write_triangle_mesh(str(combined_path), combined)
    log.info("[%s] combined mesh → %s", name, combined_path)

    if cfg.get("visualize", False):
        vis_path = out_dir / f"{name}_visualization.html"
        visualization.visualize(segments, mesh_records, vis_path)
        log.info("[%s] visualization → %s", name, vis_path)

    overall = evaluation.evaluate(xyz, combined)
    return {"name": name, "segments": seg_metrics, "combined": overall}
