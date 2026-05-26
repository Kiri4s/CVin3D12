"""Load .ply files with x/y/z + scalar_Label."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _parse_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz [N,3], labels [N]) from an ASCII .ply with scalar_Label."""
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    # Find end_header and property order
    props: list[str] = []
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("property"):
            props.append(stripped.split()[-1])
        if stripped == "end_header":
            header_end = i + 1
            break

    if not props or header_end == 0:
        raise ValueError(f"Invalid PLY header in {path}")

    xi = props.index("x")
    yi = props.index("y")
    zi = props.index("z")
    li = props.index("scalar_Label")

    data = np.loadtxt(lines[header_end:], dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] == 0:
        raise ValueError(f"Empty point cloud in {path}")

    xyz = data[:, [xi, yi, zi]]
    labels = data[:, li].astype(int)

    nan_mask = np.any(np.isnan(xyz), axis=1)
    if nan_mask.any():
        log.warning("%s: dropping %d NaN points", path.name, nan_mask.sum())
        xyz = xyz[~nan_mask]
        labels = labels[~nan_mask]

    return xyz, labels


def load_dataset(dataset_dir: str) -> list[dict]:
    """Load all .ply files from *dataset_dir*. Returns list of cloud dicts."""
    root = Path(dataset_dir)
    clouds = []
    for ply_path in sorted(root.glob("*.ply")):
        try:
            xyz, labels = _parse_ply(ply_path)
            clouds.append({"name": ply_path.stem, "xyz": xyz, "labels": labels})
        except Exception as exc:
            log.error("Skipping %s: %s", ply_path.name, exc)
    return clouds
