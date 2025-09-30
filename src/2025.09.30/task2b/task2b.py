from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import logging

import hydra
from omegaconf import OmegaConf


@dataclass
class BBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@dataclass
class Config:
    # Input / output
    input_path: str
    output_dir: str

    # Filters
    bbox: BBox
    height_threshold: float
    distance_center: Tuple[float, float, float]
    distance_radius: float
    class_filter: int  # set >=0 to filter by class, -1 means ignored

    # Visualization / misc
    viz: bool
    downsample_for_viz: int  # points to plot at most (random sample)
    random_seed: int


def load_point_cloud(path: str) -> np.ndarray:
    """Load .asc/.xyz file. Accepts 3 or 4 columns (x y z [class]).
    Returns Nx4 array where 4th column is integer class (0 if absent).
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 3:
        classes = np.zeros((data.shape[0], 1), dtype=int)
        data = np.concatenate([data, classes], axis=1)
    elif data.shape[1] >= 4:
        data = data[:, :4]

        data[:, 3] = data[:, 3].astype(int)
    else:
        raise ValueError("Input file must have at least 3 columns: x y z")

    return data.astype(float)


def save_point_cloud(path: str, points: np.ndarray):
    """Save Nx4 array to ASCII with format: x y z class"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pts = np.column_stack((points[:, :3], points[:, 3].astype(int)))
    fmt = "%.6f %.6f %.6f %d"
    np.savetxt(path, pts, fmt=fmt)


def filter_by_bbox(points: np.ndarray, bbox: BBox) -> np.ndarray:
    mask = (
        (points[:, 0] >= bbox.xmin)
        & (points[:, 0] <= bbox.xmax)
        & (points[:, 1] >= bbox.ymin)
        & (points[:, 1] <= bbox.ymax)
        & (points[:, 2] >= bbox.zmin)
        & (points[:, 2] <= bbox.zmax)
    )
    return points[mask]


def filter_by_height(points: np.ndarray, z_threshold: float) -> np.ndarray:
    return points[points[:, 2] > z_threshold]


def filter_by_distance(
    points: np.ndarray, center: Tuple[float, float, float], radius: float
) -> np.ndarray:
    center = np.array(center)
    distances = np.linalg.norm(points[:, :3] - center.reshape(1, 3), axis=1)
    return points[distances <= radius]


def filter_by_class(points: np.ndarray, class_id: int) -> np.ndarray:
    return points[points[:, 3] == class_id]


def downsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points or max_points <= 0:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def show_cloud(
    points: np.ndarray,
    title: str = "Point Cloud",
    save_to: str = None,
    max_points: int = 10000,
    seed: int = 0,
):
    pts = downsample(points, max_points, seed)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    classes = pts[:, 3]
    unique = np.unique(classes)
    if unique.size > 1:
        im = ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], s=1, c=classes)
        fig.colorbar(im, ax=ax, label="class")
    else:
        ax.scatter(pts[:, 1], pts[:, 0], pts[:, 2], s=0.5)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=300)
    plt.close(fig)


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("segment_pc")

    # Print config summary
    log.info("Configuration:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    in_path = Path(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        log.error("Input file does not exist: %s", in_path)
        return

    points = load_point_cloud(str(in_path))
    log.info("Loaded point cloud: %d points", points.shape[0])

    # 2. Bounding box
    bbox_filtered = filter_by_bbox(points, cfg.bbox)
    log.info("BBox filtered: %d points", bbox_filtered.shape[0])
    save_point_cloud(out_dir / "bbox_filtered.asc", bbox_filtered)

    # 3. Height (Z) filter
    high_points = filter_by_height(points, cfg.height_threshold)
    log.info(
        "High points (Z > %s): %d points", cfg.height_threshold, high_points.shape[0]
    )
    save_point_cloud(out_dir / "high_points.asc", high_points)

    # 4. Distance to center
    near_center = filter_by_distance(
        points, tuple(cfg.distance_center), cfg.distance_radius
    )
    log.info(
        "Near center (radius=%s): %d points", cfg.distance_radius, near_center.shape[0]
    )
    save_point_cloud(out_dir / "near_center.asc", near_center)

    # Optional: class filter
    if cfg.class_filter >= 0:
        class_pts = filter_by_class(points, cfg.class_filter)
        log.info("Class %d filtered: %d points", cfg.class_filter, class_pts.shape[0])
        save_point_cloud(out_dir / f"class_{cfg.class_filter}.asc", class_pts)

    # 6. Visualization
    if cfg.viz:
        log.info("Generating visualizations (may downsample for speed)")
        show_cloud(
            points,
            title="Original Cloud",
            save_to=str(out_dir / "viz_original.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )
        show_cloud(
            bbox_filtered,
            title="BBox Filtered",
            save_to=str(out_dir / "viz_bbox.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )
        show_cloud(
            high_points,
            title="High Points",
            save_to=str(out_dir / "viz_high.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )
        show_cloud(
            near_center,
            title="Near Center",
            save_to=str(out_dir / "viz_near_center.png"),
            max_points=cfg.downsample_for_viz,
            seed=cfg.random_seed,
        )

    # Summary text report
    report_lines = [
        f"Original points: {points.shape[0]}",
        f"BBox filtered: {bbox_filtered.shape[0]}",
        f"High points: {high_points.shape[0]}",
        f"Near center: {near_center.shape[0]}",
    ]
    (out_dir / "report.txt").write_text("\n".join(report_lines))
    log.info("Saved report and outputs to %s", out_dir)


if __name__ == "__main__":
    main()
