from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

import hydra
from omegaconf import DictConfig


@dataclass
class Config:
    # I/O
    input_path: str
    output_dir: str

    # Subsampling options
    method: str
    n_samples: int
    voxel_size: float

    # Misc
    seed: int
    visualize: bool
    save_vis_to: str
    save_xyz: bool
    fps_init_index: int | None


def load_asc(path: str) -> np.ndarray:
    """Load a point cloud in ASCII format. Accepts 3 or 4 columns: x y z [class].

    Returns an NxC numpy array (C == 3 or 4).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try loading; skip empty/comment lines automatically
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] not in (3, 4):
        raise ValueError(
            f"Unsupported number of columns: {data.shape[1]}; expected 3 or 4"
        )

    return data.astype(float)


def save_xyz(path: str | Path, points: np.ndarray) -> None:
    """Save points to an ASCII .xyz/.asc file with either 3 or 4 columns.

    If points has 4 columns, order is x y z class.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%.6f %.6f %.6f" if points.shape[1] == 3 else "%.6f %.6f %.6f %.6f"
    np.savetxt(str(path), points, fmt=fmt)


def random_subsampling(
    points: np.ndarray, n_samples: int, seed: int | None = None
) -> np.ndarray:
    """Pick n_samples uniformly at random (without replacement).

    points: (N,3) or (N,4)
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if n_samples >= N:
        return points.copy()
    idx = rng.choice(N, size=n_samples, replace=False)
    return points[idx]


def voxel_grid_subsampling(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Simple voxel-grid subsampling.

    - Compute integer voxel indices by floor(coord / voxel_size).
    - Use np.unique on rows of voxel indices to pick the first point that falls into each voxel.

    Returns subsampled points with same columns as input.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    # unique rows and keep first occurrence
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def farthest_point_sampling(
    points: np.ndarray, k: int, init_index: int | None = None
) -> np.ndarray:
    """Farthest Point Sampling (FPS) implemented with pure NumPy.

    Complexity: O(N * k). Works for moderate N (e.g. up to a few 100k depending on k).

    points: (N, >=3) array. Only x,y,z are considered.
    k: number of points to sample
    init_index: optional starting index; if None, choose 0
    """
    N = points.shape[0]
    if k >= N:
        return points.copy()
    xyz = points[:, :3]

    if init_index is None:
        init = 0
    else:
        init = int(init_index)

    chosen_idx = np.empty(k, dtype=np.int64)
    chosen_idx[0] = init

    # distances to nearest chosen point (init as +inf except for first)
    dist2 = np.full(N, np.inf)

    # Precompute squared distances iteratively
    last_chosen = xyz[init]
    # Update distances after first choice
    diff = xyz - last_chosen
    dist2 = np.minimum(dist2, np.sum(diff * diff, axis=1))

    for i in range(1, k):
        # pick farthest point
        next_idx = int(np.argmax(dist2))
        chosen_idx[i] = next_idx
        last_chosen = xyz[next_idx]
        # update distances
        d = xyz - last_chosen
        dist2 = np.minimum(dist2, np.sum(d * d, axis=1))

    return points[chosen_idx]


def visualize_point_cloud(
    points: np.ndarray,
    title: str = "Point Cloud",
    s: float = 1.0,
    save_to: str = "./cloud.png",
) -> None:
    """Simple 3D scatter using matplotlib. Colors by class column if present, else by z."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # x-axis and y-axis swapped for better view
    xyz = points[:, :3]
    if points.shape[1] == 4:
        c = points[:, 3]
        im = ax.scatter(xyz[:, 1], xyz[:, 0], xyz[:, 2], s=s, c=c)
        fig.colorbar(im, ax=ax, label="class")
    else:
        im = ax.scatter(xyz[:, 1], xyz[:, 0], xyz[:, 2], s=s)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_to)


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("subsample")

    start_all = time.time()
    input_path = Path(cfg.input_path)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading: {input_path}")
    t0 = time.time()
    pts = load_asc(str(input_path))
    t1 = time.time()
    log.info(f"Loaded {pts.shape[0]} points in {t1 - t0:.3f} s")

    method = cfg.method.lower()
    np.random.seed(cfg.seed)

    if method == "random":
        t0 = time.time()
        out = random_subsampling(pts, cfg.n_samples, seed=cfg.seed)
        t1 = time.time()
        log.info(f"Random subsampling => {out.shape[0]} pts, took {t1 - t0:.3f} s")
        out_name = out_dir / f"subsampled_random_{out.shape[0]}.xyz"

    elif method == "voxel":
        t0 = time.time()
        out = voxel_grid_subsampling(pts, cfg.voxel_size)
        t1 = time.time()
        log.info(
            f"Voxel-grid subsampling (voxel_size={cfg.voxel_size}) => {out.shape[0]} pts, took {t1 - t0:.3f} s"
        )
        out_name = out_dir / f"subsampled_voxel_{cfg.voxel_size:.3f}_{out.shape[0]}.xyz"

    elif method == "fps":
        t0 = time.time()
        init_idx = cfg.fps_init_index
        out = farthest_point_sampling(pts, cfg.n_samples, init_index=init_idx)
        t1 = time.time()
        log.info(f"FPS subsampling => {out.shape[0]} pts, took {t1 - t0:.3f} s")
        out_name = out_dir / f"subsampled_fps_{out.shape[0]}.xyz"

    else:
        raise ValueError(f"Unknown method: {method}. Choose from random | voxel | fps")

    if cfg.save_xyz:
        save_xyz(out_name, out)
        log.info(f"Saved subsampled cloud to: {out_name}")

    if cfg.visualize:
        # choose marker size adaptively
        s = max(0.1, 20_000.0 / max(out.shape[0], 1))
        visualize_point_cloud(
            out,
            title=f"Subsampled ({method}) - {out.shape[0]} pts",
            s=s,
            save_to=cfg.save_vis_to,
        )

    log.info(f"Total time: {time.time() - start_all:.3f} s")


if __name__ == "__main__":
    main()
