import numpy as np
import hydra
from utils import load_point_cloud
from pathlib import Path
from typing import Tuple, List


def rotate_x(angle: float) -> np.ndarray:
    """
    Generate a 3D rotation matrix for rotation around the X-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix for X-axis rotation.
    """
    rot_matrix = np.eye(3)
    rot_matrix[1, 1] = np.cos(angle)
    rot_matrix[1, 2] = -np.sin(angle)
    rot_matrix[2, 2] = np.cos(angle)
    rot_matrix[2, 1] = np.sin(angle)

    return rot_matrix


def rotate_y(angle: float) -> np.ndarray:
    """
    Generate a 3D rotation matrix for rotation around the Y-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix for Y-axis rotation.
    """
    rot_matrix = np.eye(3)
    rot_matrix[0, 0] = np.cos(angle)
    rot_matrix[0, 2] = np.sin(angle)
    rot_matrix[2, 2] = np.cos(angle)
    rot_matrix[2, 0] = -np.sin(angle)

    return rot_matrix


def rotate_z(angle: float) -> np.ndarray:
    """
    Generate a 3D rotation matrix for rotation around the Z-axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        A 3x3 rotation matrix for Z-axis rotation.
    """
    rot_matrix = np.eye(3)
    rot_matrix[0, 0] = np.cos(angle)
    rot_matrix[0, 1] = -np.sin(angle)
    rot_matrix[1, 1] = np.cos(angle)
    rot_matrix[1, 0] = np.sin(angle)

    return rot_matrix


def rotate_by_angles(
    angles: Tuple | List, axes: str, coords: np.ndarray, inplace=True
) -> np.ndarray:
    """
    Apply sequential rotations to 3D coordinates around specified axes.

    Args:
        angles: Sequence of rotation angles in radians.
        axes: String containing axis names ('x', 'y', 'z') for each rotation.
        coords: Array of 3D coordinates with shape (N, 3).
        inplace: If True, modify the input array in-place; otherwise return a new array.

    Returns:
        Rotated coordinates array.

    Raises:
        ValueError: If axes string is empty, contains invalid characters, or length doesn't match angles.
    """
    axes = axes.strip().lower()
    if len(angles) == 0:
        return coords
    if len(axes) == 0:
        raise ValueError("Axes string is empty")
    for c in axes:
        if c not in "xyz":
            raise ValueError(f"Unknown rotation axis: {c}")
    if len(angles) != len(axes):
        raise ValueError("Number of angles and axes must match")

    rot_matrix = np.eye(3)
    for angle, axis in zip(angles, axes):
        if axis == "x":
            rot_matrix = rotate_x(angle) @ rot_matrix
        elif axis == "y":
            rot_matrix = rotate_y(angle) @ rot_matrix
        elif axis == "z":
            rot_matrix = rotate_z(angle) @ rot_matrix

    if inplace:
        coords[:] = coords @ rot_matrix.T
        return coords

    return coords @ rot_matrix.T


def add_gaussian_noise(coords: np.ndarray, stddev: float, inplace=True) -> np.ndarray:
    """
    Add Gaussian (normal) noise to 3D coordinates.

    Args:
        coords: Array of 3D coordinates with shape (N, 3).
        stddev: Standard deviation of the Gaussian noise.
        inplace: If True, modify the input array in-place; otherwise return a new array.

    Returns:
        Coordinates with added Gaussian noise.
    """
    noise = np.random.normal(loc=0.0, scale=stddev, size=coords.shape).astype(
        np.float32
    )
    if inplace:
        coords += noise
        return coords
    return coords + noise


def add_uniform_noise(
    coords: np.ndarray, low: float, high: float, inplace=True
) -> np.ndarray:
    """
    Add uniform random noise to 3D coordinates.

    Args:
        coords: Array of 3D coordinates with shape (N, 3).
        low: Lower bound of the uniform noise range.
        high: Upper bound of the uniform noise range.
        inplace: If True, modify the input array in-place; otherwise return a new array.

    Returns:
        Coordinates with added uniform noise.
    """
    noise = np.random.uniform(low=low, high=high, size=coords.shape).astype(np.float32)
    if inplace:
        coords += noise
        return coords
    return coords + noise


@hydra.main(config_path=".", config_name="conf.yaml", version_base=None)
def main(cfg) -> None:
    input_txt = Path(cfg.input_txt)

    coords = load_point_cloud(input_txt, max_rows=1000)[:, :3]

    if cfg.rotation.apply:
        print("Before rotation:")
        print(coords[:5])
        rotate_by_angles(
            cfg.rotation.angles, cfg.rotation.axes, coords, inplace=cfg.rotation.inplace
        )
        print("After rotation:")
        print(coords[:5], "\n")

    if cfg.gaussian_noise.apply:
        print("Before Gaussian noise:")
        print(coords[:5])
        add_gaussian_noise(
            coords, cfg.gaussian_noise.std, inplace=cfg.gaussian_noise.inplace
        )
        print("After Gaussian noise:")
        print(coords[:5], "\n")

    if cfg.uniform_noise.apply:
        print("Before Uniform noise:")
        print(coords[:5])
        add_uniform_noise(
            coords,
            cfg.uniform_noise.low,
            cfg.uniform_noise.high,
            inplace=cfg.uniform_noise.inplace,
        )
        print("After Uniform noise:")
        print(coords[:5], "\n")


if __name__ == "__main__":
    main()
