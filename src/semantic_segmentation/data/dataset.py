from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Column order in PLY data section:
# x y z label instance_id red green blue station_index circle_index elevation_deg
_COLS = ["x", "y", "z", "label", "instance_id", "red", "green", "blue",
         "station_index", "circle_index", "elevation_deg"]
_USE_COLS = ["x", "y", "z", "label", "red", "green", "blue"]


def _count_header_lines(path: str) -> int:
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            if line.strip() == "end_header":
                return i
    return 0


def load_ply(path: str):
    skip = _count_header_lines(path)
    data = pd.read_csv(
        path, sep=r"\s+", skiprows=skip, header=None,
        names=_COLS, usecols=_USE_COLS,
    )
    xyz = data[["x", "y", "z"]].to_numpy(dtype=np.float32)
    rgb = data[["red", "green", "blue"]].to_numpy(dtype=np.float32) / 255.0
    labels = data["label"].to_numpy(dtype=np.int64)
    return xyz, rgb, labels


class PointCloudDataset(Dataset):
    def __init__(self, scene_dirs: list, num_points: int, samples_per_cloud: int):
        self.num_points = num_points
        self._cache: dict = {}

        files = []
        for d in scene_dirs:
            files.extend(sorted(Path(d).rglob("*.ply")))

        self.index = [(str(f), i) for f in files for i in range(samples_per_cloud)]

    def _get_cloud(self, path: str):
        if path not in self._cache:
            self._cache[path] = load_ply(path)
        return self._cache[path]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        path, _ = self.index[idx]
        xyz, rgb, labels = self._get_cloud(path)

        n = len(labels)
        sel = np.random.choice(n, self.num_points, replace=n < self.num_points)

        xyz_s = xyz[sel].copy()
        xyz_s -= xyz_s.mean(0)
        xyz_s /= np.abs(xyz_s).max() + 1e-8

        features = np.concatenate([xyz_s, rgb[sel]], axis=1).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(labels[sel])
