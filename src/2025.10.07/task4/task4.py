from dataclasses import dataclass
import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

from hydra import main
from hydra.utils import get_original_cwd
from omegaconf import MISSING
from pathlib import Path


@dataclass
class Config:
    # IO
    input_path: str = "data/cloud.txt"
    output_dir: str = "outputs"

    # Task parameters
    constant_value: float = 10.0
    multiply_factor: float = 2.0
    add_value: float = 5.0

    gaussian_sigma: float = 2.0
    moving_average_window: int = 5

    colormap: str = "viridis"

    filter_min: float = -1e9
    filter_max: float = 1e9

    use_scalar_axis: int = 2  # replace Z by scalar by default


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def load_points(path: str) -> np.ndarray:
    """
    Load point cloud from .npy, .txt, .asc, or .csv. Expect Nx3 or Nx>=3 (xyz + optional scalars).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
    elif ext in [".txt", ".csv", ".asc"]:
        data = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use .npy, .txt, or .csv.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_asc(path: str) -> np.ndarray:
    """Load a point cloud in ASCII format. Accepts 3 or 4 columns: x y z [class].

    Returns an NxC numpy array (C == 3 or 4).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try loading; skip empty/comment lines automatically
    data = np.loadtxt(str(path))

    if data.shape[1] not in (3, 4):
        raise ValueError(
            f"Unsupported number of columns: {data.shape[1]}; expected 3 or 4"
        )

    return data.astype(float)


def save_array(path: str, arr: np.ndarray):
    np.save(path, arr)


def save_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)


def make_scalar_constant(n_points: int, value: float) -> np.ndarray:
    return np.full(n_points, float(value), dtype=float)


def multiply_scalar(scalar: np.ndarray, factor: float) -> np.ndarray:
    return scalar * factor


def add_scalar_constant(scalar: np.ndarray, add_value: float) -> np.ndarray:
    return scalar + add_value


def gaussian_smooth(scalar: np.ndarray, sigma: float) -> np.ndarray:
    # using 1D gaussian filter as suggested in the assignment
    return gaussian_filter1d(scalar, sigma=float(sigma), mode="nearest")


def moving_average(scalar: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size <= 1:
        return scalar.copy()
    kernel = np.ones(window_size) / float(window_size)
    # mode='same' behaviour via np.convolve
    return np.convolve(scalar, kernel, mode="same")


def compute_gradient(
    scalar: np.ndarray, coords: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute gradient of scalar field.
    If coords (Nx1 or Nx) provided, attempt to compute gradient along sorted coordinates.
    Otherwise use numpy.gradient with unit spacing.
    """
    if coords is None:
        return np.gradient(scalar)
    coords = np.asarray(coords).flatten()
    # sort by coordinate to get a meaningful 1D gradient
    idx = np.argsort(coords)
    sorted_coords = coords[idx]
    sorted_scalar = scalar[idx]
    grad_sorted = np.gradient(sorted_scalar, sorted_coords)
    # return gradient in original order
    grad = np.empty_like(grad_sorted)
    grad[idx] = grad_sorted
    return grad


def interpolate_nan(scalar: np.ndarray) -> np.ndarray:
    x = np.arange(len(scalar))
    nans = np.isnan(scalar)
    if not np.any(nans):
        return scalar.copy()
    # If all NaN -> fill zero
    if np.all(nans):
        return np.zeros_like(scalar)
    interp_func = interp1d(
        x[~nans], scalar[~nans], bounds_error=False, fill_value="extrapolate"
    )
    return interp_func(x)


def normalize_to_unit(scalar: np.ndarray, clip: bool = True) -> np.ndarray:
    minv = np.nanmin(scalar)
    maxv = np.nanmax(scalar)
    if np.isclose(maxv, minv):
        # avoid division by zero
        return np.zeros_like(scalar, dtype=float)
    out = (scalar - minv) / (maxv - minv)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def scalar_to_rgb(scalar: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """
    Returns Nx3 array of RGB in range [0,1].
    """
    import matplotlib.cm as cm

    normed = normalize_to_unit(scalar)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(normed)
    return rgba[:, :3]


def filter_points_by_scalar(
    points: np.ndarray, scalar: np.ndarray, min_v: float, max_v: float
) -> np.ndarray:
    mask = (scalar >= min_v) & (scalar <= max_v)
    return points[mask], scalar[mask]


def use_scalar_as_coordinate(
    points: np.ndarray, scalar: np.ndarray, axis: int = 2
) -> np.ndarray:
    """
    Return new points array where coordinate 'axis' (0=X,1=Y,2=Z) is replaced by scalar.
    """
    pts = points.copy()
    if pts.shape[1] < 3:
        raise ValueError("Points must have at least 3 columns (xyz).")
    pts[:, axis] = scalar
    return pts


def remove_scalar_field(
    points_with_scalar: np.ndarray, scalar_index: int = 3
) -> np.ndarray:
    """
    Remove the scalar column at scalar_index; default assumes columns are [x,y,z,scalar,...].
    """
    if points_with_scalar.shape[1] <= scalar_index:
        # nothing to remove
        return points_with_scalar
    return np.delete(points_with_scalar, scalar_index, axis=1)


def compute_stats(scalar: np.ndarray) -> Dict[str, float]:
    return {
        "count": float(np.sum(~np.isnan(scalar))),
        "mean": float(np.nanmean(scalar)),
        "std": float(np.nanstd(scalar)),
        "min": float(np.nanmin(scalar)),
        "max": float(np.nanmax(scalar)),
        "median": float(np.nanmedian(scalar)),
    }


def plot_3d_colored(
    points: np.ndarray, scalar: np.ndarray, cmap: str, out_path: str, title: str = ""
):
    """
    Save a 3D scatter (matplotlib) colored by scalar.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    normed = normalize_to_unit(scalar)
    p = ax.scatter(points[:, 1], points[:, 0], points[:, 2], c=normed, cmap=cmap, s=1)
    ax.set_title(title)
    fig.colorbar(p, ax=ax, label="normalized scalar")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scalar_line(
    scalar: np.ndarray, out_path: str, title: str = "Scalar field (1D view)"
):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(scalar, linewidth=1)
    ax.set_title(title)
    ax.grid()
    ax.set_xlabel("point index")
    ax.set_ylabel("scalar value")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@main(version_base=None, config_name="config.yaml", config_path=".")
def main(cfg: Config) -> None:
    # hydra changes working directory; resolve output dir relative to repo root
    project_root = get_original_cwd()
    out_root = os.path.join(project_root, cfg.output_dir)
    ensure_dir(out_root)

    # Load points
    pts = load_asc(os.path.join(project_root, cfg.input_path))
    n_points = pts.shape[0]
    xyz = pts[:, :3]
    # If file had an existing scalar (column 3), use it; else create primary scalar of zeros
    scalar = None
    if pts.shape[1] >= 4:
        scalar = pts[:, 3].astype(float)
    else:
        # If no scalar present, create constant scalar as starting value (task 1)
        scalar = make_scalar_constant(n_points, cfg.constant_value)

    # Save initial data
    save_array(os.path.join(out_root, "xyz.npy"), xyz)
    save_array(os.path.join(out_root, "scalar_initial.npy"), scalar)

    # --- Task 1: Add scalar with constant value (if not provided) ---
    scalar_t1 = make_scalar_constant(n_points, cfg.constant_value)
    save_array(os.path.join(out_root, "scalar_task1_constant.npy"), scalar_t1)
    plot_scalar_line(
        scalar_t1,
        os.path.join(out_root, "scalar_task1_constant.png"),
        "Task1: constant scalar",
    )

    # --- Task 2: Multiply scalar by factor ---
    scalar_t2 = multiply_scalar(scalar, cfg.multiply_factor)
    save_array(os.path.join(out_root, "scalar_task2_multiplied.npy"), scalar_t2)
    plot_scalar_line(
        scalar_t2,
        os.path.join(out_root, "scalar_task2_multiplied.png"),
        "Task2: multiplied scalar",
    )

    # --- Task 3: Add number to scalar values ---
    scalar_t3 = add_scalar_constant(scalar_t2, cfg.add_value)
    save_array(os.path.join(out_root, "scalar_task3_added.npy"), scalar_t3)
    plot_scalar_line(
        scalar_t3,
        os.path.join(out_root, "scalar_task3_added.png"),
        "Task3: added scalar",
    )

    # --- Task 4: Gaussian smoothing ---
    scalar_t4 = gaussian_smooth(scalar_t3, cfg.gaussian_sigma)
    save_array(os.path.join(out_root, "scalar_task4_gaussian.npy"), scalar_t4)
    plot_scalar_line(
        scalar_t4,
        os.path.join(out_root, "scalar_task4_gaussian.png"),
        f"Task4: gaussian sigma={cfg.gaussian_sigma}",
    )

    # --- Task 5: Gradient ---
    # We'll compute gradient in index order; if points are ordered along a coordinate user could pass it here
    gradient = compute_gradient(scalar_t4, coords=None)
    save_array(os.path.join(out_root, "scalar_task5_gradient.npy"), gradient)

    # --- Task 6: Moving average (re-smoothing) ---
    scalar_t6 = moving_average(scalar_t4, cfg.moving_average_window)
    save_array(os.path.join(out_root, "scalar_task6_moving_avg.npy"), scalar_t6)
    plot_scalar_line(
        scalar_t6,
        os.path.join(out_root, "scalar_task6_moving_avg.png"),
        f"Task6: moving average window={cfg.moving_average_window}",
    )

    # --- Task 7: Scalar -> RGB ---
    colors = scalar_to_rgb(scalar_t6, cmap_name=cfg.colormap)
    save_array(os.path.join(out_root, "colors_task7_rgb.npy"), colors)
    # 3D colored scatter (matplotlib)
    plot_3d_colored(
        xyz,
        scalar_t6,
        cfg.colormap,
        os.path.join(out_root, "task7_colored_scatter.png"),
        title="Task7: colored by scalar",
    )

    # --- Task 8: Statistics (table) ---
    stats = compute_stats(scalar_t6)
    df_stats = pd.DataFrame([stats])
    save_csv(os.path.join(out_root, "task8_stats.csv"), df_stats)
    # also save a human-readable small table image using matplotlib
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_stats.round(5).values, colLabels=df_stats.columns, loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_root, "task8_stats_table.png"), dpi=150)
    plt.close(fig)

    # --- Task 9: Normalize to [0,1] ---
    scalar_t9 = normalize_to_unit(scalar_t6)
    save_array(os.path.join(out_root, "scalar_task9_normalized.npy"), scalar_t9)
    plot_scalar_line(
        scalar_t9,
        os.path.join(out_root, "scalar_task9_normalized.png"),
        "Task9: normalized scalar",
    )

    # --- Task 10: Interpolate NaNs (simulate some NaNs if none) ---
    scalar_with_nans = scalar_t9.copy()
    # For demonstration, set a few random values to NaN if none present
    if not np.any(np.isnan(scalar_with_nans)):
        rng = np.random.default_rng(42)
        idxs = rng.choice(
            len(scalar_with_nans),
            size=max(1, len(scalar_with_nans) // 50),
            replace=False,
        )
        scalar_with_nans[idxs] = np.nan
    scalar_t10 = interpolate_nan(scalar_with_nans)
    save_array(os.path.join(out_root, "scalar_task10_interpolated.npy"), scalar_t10)
    plot_scalar_line(
        scalar_with_nans,
        os.path.join(out_root, "scalar_task10_with_nans.png"),
        "Task10: with NaNs",
    )
    plot_scalar_line(
        scalar_t10,
        os.path.join(out_root, "scalar_task10_interpolated.png"),
        "Task10: interpolated",
    )

    # --- Task 11: Filter by scalar value ---
    min_v = cfg.filter_min
    max_v = cfg.filter_max
    filtered_pts, filtered_scalar = filter_points_by_scalar(
        xyz, scalar_t10, min_v, max_v
    )
    save_array(os.path.join(out_root, "task11_filtered_xyz.npy"), filtered_pts)
    save_array(os.path.join(out_root, "task11_filtered_scalar.npy"), filtered_scalar)
    if filtered_pts.shape[0] > 0:
        plot_3d_colored(
            filtered_pts,
            filtered_scalar,
            cfg.colormap,
            os.path.join(out_root, "task11_filtered_scatter.png"),
            title=f"Task11: filtered [{min_v},{max_v}]",
        )

    # --- Task 12: Use scalar as coordinate (e.g., replace Z) ---
    used_pts = use_scalar_as_coordinate(xyz, scalar_t10, axis=cfg.use_scalar_axis)
    save_array(os.path.join(out_root, "task12_used_scalar_as_coord.npy"), used_pts)
    plot_3d_colored(
        used_pts,
        scalar_t10,
        cfg.colormap,
        os.path.join(out_root, "task12_scalar_as_coord.png"),
        title="Task12: scalar used as coordinate",
    )

    # --- Task 13: Remove scalar field from points (if points had one) ---
    if pts.shape[1] >= 4:
        pts_removed = remove_scalar_field(pts, scalar_index=3)
        save_array(
            os.path.join(out_root, "task13_removed_scalar_points.npy"), pts_removed
        )

    # Summarize saved files
    summary = {
        "n_points": int(n_points),
        "outputs_dir": out_root,
        "saved_files": os.listdir(out_root),
    }

    # Save summary as json/csv
    pd.DataFrame([summary]).to_json(
        os.path.join(out_root, "run_summary.json"), orient="records", indent=2
    )

    print("Pipeline finished. Outputs written to:", out_root)
    print("Key output files:", summary["saved_files"])


if __name__ == "__main__":
    main()
