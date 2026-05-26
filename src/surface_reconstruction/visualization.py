"""Interactive 3-layer HTML visualization: point cloud / segment mesh / method mesh."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

_PALETTE = pc.qualitative.Plotly  # 10 colours, cycled by segment index
_METHOD_COLOR = {
    "poisson":       "#636EFA",
    "alpha_shapes":  "#00CC96",
    "ball_pivoting": "#EF553B",
}


def visualize(
    segments: list[dict],
    mesh_records: list[dict],
    out_path: Path,
) -> None:
    """Build interactive HTML with 3 switchable layers and write to *out_path*.

    segments     — list of {"label": int, "xyz": ndarray}
    mesh_records — list of {"label": int, "method": str, "mesh": o3d.TriangleMesh}
    """
    n = len(mesh_records)

    def seg_color(i: int) -> str:
        return _PALETTE[i % len(_PALETTE)]

    # --- Layer 1: point cloud (one Scatter3d per segment) ---
    cloud_traces: list[go.BaseTraceType] = []
    for i, seg in enumerate(segments):
        pts = seg["xyz"]
        cloud_traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=seg_color(i)),
            name=f"label {seg['label']}",
            visible=True,
        ))

    # --- Layers 2 & 3: one Mesh3d per segment (two parallel sets) ---
    seg_mesh_traces: list[go.BaseTraceType] = []
    method_mesh_traces: list[go.BaseTraceType] = []
    for i, rec in enumerate(mesh_records):
        verts = np.asarray(rec["mesh"].vertices)
        tris  = np.asarray(rec["mesh"].triangles)
        common = dict(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=tris[:, 0],  j=tris[:, 1],  k=tris[:, 2],
            opacity=0.85, flatshading=False,
        )
        seg_mesh_traces.append(go.Mesh3d(
            **common,
            color=seg_color(i),
            name=f"label {rec['label']}",
            visible=False,
        ))
        method_mesh_traces.append(go.Mesh3d(
            **common,
            color=_METHOD_COLOR.get(rec["method"], "#888888"),
            name=rec["method"],
            visible=False,
        ))

    all_traces = cloud_traces + seg_mesh_traces + method_mesh_traces
    nc, nm = len(cloud_traces), n

    def vis(show_cloud: bool, show_seg: bool, show_method: bool) -> list[bool]:
        return ([show_cloud] * nc + [show_seg] * nm + [show_method] * nm)

    buttons = [
        dict(label="Облако",   method="update", args=[{"visible": vis(True,  False, False)}]),
        dict(label="Сегменты", method="update", args=[{"visible": vis(False, True,  False)}]),
        dict(label="Методы",   method="update", args=[{"visible": vis(False, False, True)}]),
    ]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", buttons=buttons,
            direction="right", x=0.0, y=1.08,
            showactive=True,
        )],
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing="constant"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
