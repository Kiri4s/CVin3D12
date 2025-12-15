import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def load_ply_file(file_path):
    """Load .ply file and return vertices and labels"""
    with open(file_path, "r") as f:
        # Read header to find end of header
        line = f.readline()
        while line:
            if line.strip() == "end_header":
                break
            line = f.readline()

        # Read vertex data (x, y, z, label)
        verts = []
        labels = []
        for line in f:
            values = line.strip().split()
            if len(values) >= 4:  # x, y, z, label
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                label = (
                    int(values[3]) if len(values) > 3 else 0
                )  # default label 0 if not provided
                verts.append([x, y, z])
                labels.append(label)
            elif len(values) == 3:  # x, y, z only
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                verts.append([x, y, z])

        return np.array(verts), np.array(labels)


def sample_point_cloud(vertices, labels, num_points=1024):
    """Sample points from mesh vertices"""
    if num_points is None:
        return vertices, labels

    if len(vertices) < num_points:
        indices = np.random.choice(len(vertices), num_points, replace=True)
    else:
        indices = np.random.choice(len(vertices), num_points, replace=False)

    point_cloud = vertices[indices]
    if len(labels) > 0:
        point_labels = labels[indices]
    return point_cloud, point_labels


def normalize_point_cloud(point_cloud):
    """Normalize point cloud to [-1, 1]"""
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid

    # Scale to [-1, 1]
    max_dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    point_cloud = point_cloud / max_dist

    return point_cloud


class ValveSegmentationDataset(Dataset):
    def __init__(self, root, num_points=1024, split="train", val_sz=0.1, test_sz=0.1):
        self.root = root
        self.num_points = num_points
        self.split = split

        self.clouds = np.array(os.listdir(root))
        train_idx = np.random.choice(
            len(self.clouds),
            int((1 - val_sz - test_sz) * len(self.clouds)),
            replace=False,
        )
        val_idx = np.random.choice(
            list(set(range(len(self.clouds))) - set(train_idx)),
            int(val_sz * len(self.clouds)),
            replace=False,
        )
        test_idx = np.array(
            list(set(range(len(self.clouds))) - set(train_idx) - set(val_idx))
        )

        if split == "full":
            self.clouds = self.clouds
        elif split == "train":
            self.clouds = self.clouds[train_idx]
        elif split == "val":
            self.clouds = self.clouds[val_idx]
        elif split == "test":
            self.clouds = self.clouds[test_idx]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        file_path = self.clouds[idx]
        vertices, labels = load_ply_file(os.path.join(self.root, file_path))

        point_cloud, point_labels = sample_point_cloud(
            vertices, labels, self.num_points
        )

        point_cloud = normalize_point_cloud(point_cloud)

        point_cloud = torch.FloatTensor(point_cloud)

        # Transpose to (3, N) for PointNet input
        point_cloud = point_cloud.transpose(0, 1)

        return point_cloud, torch.LongTensor(point_labels)


class ValveSegmentationLightningDataset(pl.LightningDataModule):
    def __init__(
        self,
        root,
        num_points=1024,
        num_classes=13,
        batch_size=2,
        num_workers=4,
        val_sz=0.1,
        test_sz=0.1,
    ):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_sz = val_sz
        self.test_sz = test_sz

    def setup(self, stage=None):
        self.train_dataset = ValveSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="train",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

        self.val_dataset = ValveSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="val",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

        self.test_dataset = ValveSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="test",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = ValveSegmentationDataset(
        root="../../../datasets/ds4segmentation", num_points=1024, split="test"
    )
    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    point_cloud, labels = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]} (first 5 labels)")
