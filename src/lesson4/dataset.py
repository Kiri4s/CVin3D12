import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import glob
from tqdm import tqdm


def load_off_file(file_path):
    """Load .off file and return vertices and faces"""
    with open(file_path, "r") as f:
        if "OFF" != f.readline().strip():
            raise ValueError("Not a valid OFF header")

        n_verts, n_faces, _ = map(int, f.readline().strip().split())

        verts = []
        for _ in range(n_verts):
            verts.append(list(map(float, f.readline().strip().split())))

        return np.array(verts)


def sample_point_cloud(vertices, num_points=1024):
    """Sample points from mesh vertices"""
    if len(vertices) < num_points:
        indices = np.random.choice(len(vertices), num_points, replace=True)
    else:
        indices = np.random.choice(len(vertices), num_points, replace=False)

    point_cloud = vertices[indices]
    return point_cloud


def normalize_point_cloud(point_cloud):
    """Normalize point cloud to [-1, 1]"""
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid

    # Scale to [-1, 1]
    max_dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    point_cloud = point_cloud / max_dist

    return point_cloud


class ModelNetDataset(Dataset):
    """
    ModelNet10 Dataset

    Args:
        root: Root directory of ModelNet dataset
        num_points: Number of points to sample from each object
        split: 'train' or 'test'
        num_classes: 10
    """

    def __init__(self, root, num_points=1024, split="train", num_classes=10):
        self.root = root
        self.num_points = num_points
        self.split = split
        self.num_classes = num_classes

        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )

        if num_classes == 10:
            assert len(self.classes) == 10, (
                f"Expected 10 classes, found {len(self.classes)}"
            )
        elif num_classes == 40:
            assert len(self.classes) == 40, (
                f"Expected 40 classes, found {len(self.classes)}"
            )

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.data_paths = []
        self.labels = []

        print(f"Loading {split} data for {num_classes} classes...")
        for cls_name in tqdm(self.classes, desc="Loading classes"):
            cls_dir = os.path.join(root, cls_name, split)

            if not os.path.exists(cls_dir):
                print(f"Warning: {cls_dir} does not exist, skipping...")
                continue

            off_files = glob.glob(os.path.join(cls_dir, "*.off"))

            for off_file in off_files:
                self.data_paths.append(off_file)
                self.labels.append(self.class_to_idx[cls_name])

        print(f"Loaded {len(self.data_paths)} {split} samples")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        vertices = load_off_file(file_path)

        point_cloud = sample_point_cloud(vertices, self.num_points)

        point_cloud = normalize_point_cloud(point_cloud)

        point_cloud = torch.FloatTensor(point_cloud)
        label = self.labels[idx]

        # Transpose to (3, N) for PointNet input
        point_cloud = point_cloud.transpose(0, 1)

        return point_cloud, label

    def get_class_name(self, idx):
        """Get class name from index"""
        return self.classes[idx]


class ModelNetLightningDataset(pl.LightningDataModule):
    def __init__(
        self, root, num_points=1024, num_classes=10, batch_size=32, num_workers=4
    ):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ModelNetDataset(
            root=self.root,
            num_points=self.num_points,
            split="train",
            num_classes=self.num_classes,
        )

        self.test_dataset = ModelNetDataset(
            root=self.root,
            num_points=self.num_points,
            split="test",
            num_classes=self.num_classes,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
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
    dataset = ModelNetDataset(
        root="./data/ModelNet10", num_points=1024, split="train", num_classes=10
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # Test loading a sample
    point_cloud, label = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Label: {label} ({dataset.get_class_name(label)})")
