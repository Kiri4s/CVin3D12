import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_ply(path):
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            data = np.array(
                [list(map(float, ln.split())) for ln in lines[i + 1 :] if ln.strip()]
            )
            break
    return data[:, :3].astype(np.float32), data[:, 3].astype(np.int64)


class ValveDataset(Dataset):
    def __init__(self, files, num_points=2048, augment=False):
        self.files = files
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        xyz, labels = load_ply(self.files[idx])
        N = len(xyz)
        choice = np.random.choice(N, self.num_points, replace=(N < self.num_points))
        xyz, labels = xyz[choice], labels[choice]

        xyz -= xyz.mean(0)
        xyz /= np.abs(xyz).max() + 1e-8

        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ R.T
            xyz += np.random.randn(*xyz.shape).astype(np.float32) * 0.02

        return torch.from_numpy(xyz), torch.from_numpy(labels)


def split_files(dataset_dir, train=0.7, val=0.15, seed=42):
    files = sorted(
        [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".ply")
        ]
    )
    np.random.seed(seed)
    idx = np.random.permutation(len(files))
    n_train = int(len(files) * train)
    n_val = int(len(files) * val)
    return (
        [files[i] for i in idx[:n_train]],
        [files[i] for i in idx[n_train : n_train + n_val]],
        [files[i] for i in idx[n_train + n_val :]],
    )
