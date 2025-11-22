import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from dataset import ModelNetDataset
from model import PointNetLightning
from tensorboard.backend.event_processing import event_accumulator


def visualize_point_cloud_2d(point_cloud, title="Point Cloud", color="b"):
    """
    Visualize point cloud in 2D (3 views)

    Args:
        point_cloud: numpy array of shape (3, N) or (N, 3)
    """
    if point_cloud.shape[0] == 3:
        point_cloud = point_cloud.T

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(point_cloud[:, 0], point_cloud[:, 1], c=color, s=1, alpha=0.6)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title(f"{title} - XY View")
    axes[0].set_aspect("equal")

    axes[1].scatter(point_cloud[:, 0], point_cloud[:, 2], c=color, s=1, alpha=0.6)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    axes[1].set_title(f"{title} - XZ View")
    axes[1].set_aspect("equal")

    axes[2].scatter(point_cloud[:, 1], point_cloud[:, 2], c=color, s=1, alpha=0.6)
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    axes[2].set_title(f"{title} - YZ View")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    return fig


def visualize_point_cloud_3d(point_cloud, title="Point Cloud", color="b"):
    """
    Visualize point cloud in 3D

    Args:
        point_cloud: numpy array of shape (3, N) or (N, 3)
    """
    if point_cloud.shape[0] == 3:
        point_cloud = point_cloud.T

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=color, s=1, alpha=0.6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    max_range = (
        np.array(
            [
                point_cloud[:, 0].max() - point_cloud[:, 0].min(),
                point_cloud[:, 1].max() - point_cloud[:, 1].min(),
                point_cloud[:, 2].max() - point_cloud[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (point_cloud[:, 0].max() + point_cloud[:, 0].min()) * 0.5
    mid_y = (point_cloud[:, 1].max() + point_cloud[:, 1].min()) * 0.5
    mid_z = (point_cloud[:, 2].max() + point_cloud[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return fig


def visualize_predictions(
    point_clouds, predictions, labels, class_names, num_samples=9
):
    """
    Visualize predictions vs ground truth

    Args:
        point_clouds: List of point clouds (N, 3)
        predictions: List of predicted class indices
        labels: List of true class indices
        class_names: List of class names
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(point_clouds))
    rows = (num_samples + 2) // 3

    fig, axes = plt.subplots(
        rows, 3, figsize=(15, 5 * rows), subplot_kw={"projection": "3d"}
    )
    axes = axes.flatten() if num_samples > 1 else [axes]

    for i in range(num_samples):
        pc = point_clouds[i]
        if pc.shape[0] == 3:
            pc = pc.T

        pred = predictions[i]
        true = labels[i]

        color = "green" if pred == true else "red"

        axes[i].scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=1, alpha=0.6)
        axes[i].set_title(
            f"True: {class_names[true]}\nPred: {class_names[pred]}", fontsize=10
        )
        axes[i].set_xlabel("X", fontsize=8)
        axes[i].set_ylabel("Y", fontsize=8)
        axes[i].set_zlabel("Z", fontsize=8)

        axes[i].grid(False)

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


def visualize_dataset_samples(dataset, num_samples=6):
    """Visualize random samples from dataset"""
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        point_cloud, label = dataset[idx]

        pc = point_cloud.numpy().T

        class_name = dataset.get_class_name(label)

        axes[i].scatter(pc[:, 0], pc[:, 1], pc[:, 2], c="blue", s=1, alpha=0.6)
        axes[i].set_title(f"Class: {class_name}", fontsize=12)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].set_zlabel("Z")
        axes[i].grid(False)

    plt.tight_layout()
    return fig


@hydra.main(config_path=".", config_name="conf.yaml", version_base=None)
def visualize_from_checkpoint(cfg: DictConfig):
    """Load model from checkpoint and visualize predictions"""

    test_dataset = ModelNetDataset(
        root=cfg.data.root,
        num_points=cfg.data.num_points,
        split="test",
        num_classes=cfg.data.num_classes,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    checkpoint_path = input("Enter checkpoint path: ")
    model = PointNetLightning.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.eval()

    all_preds = []
    all_labels = []
    all_point_clouds = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("Getting predictions...")
    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(test_loader):
            if batch_idx >= 3:
                break

            points = points.to(device)
            logits, _ = model(points)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_point_clouds.extend(points.cpu().numpy())

    print("Generating visualizations...")
    fig = visualize_predictions(
        all_point_clouds[:9],
        all_preds[:9],
        all_labels[:9],
        test_dataset.classes,
        num_samples=9,
    )

    plt.savefig("predictions_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved predictions_visualization.png")
    plt.show()


def plot_training_curves(log_dir):
    """Plot training curves from tensorboard logs"""

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    train_loss = ea.Scalars("train_loss")
    train_acc = ea.Scalars("train_acc")
    val_loss = ea.Scalars("val_loss")
    val_acc = ea.Scalars("val_acc")

    train_loss_vals = [x.value for x in train_loss]
    train_acc_vals = [x.value for x in train_acc]
    val_loss_vals = [x.value for x in val_loss]
    val_acc_vals = [x.value for x in val_acc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_loss_vals, label="Train Loss", alpha=0.7)
    ax1.plot(val_loss_vals, label="Val Loss", alpha=0.7)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_acc_vals, label="Train Accuracy", alpha=0.7)
    ax2.plot(val_acc_vals, label="Val Accuracy", alpha=0.7)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")
    plt.show()


if __name__ == "__main__":
    dataset = ModelNetDataset(
        root="../../../datasets/ModelNet10",
        num_points=1024,
        split="train",
        num_classes=10,
    )

    fig = visualize_dataset_samples(dataset, num_samples=6)
    plt.savefig("dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
