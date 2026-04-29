import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from dataset import load_ply
from models import get_model


def load_and_normalize_point_cloud(ply_path, num_points=2048):
    """
    Load point cloud from PLY file and normalize it.

    Args:
        ply_path (str): Path to PLY file
        num_points (int): Number of points to sample (if more points exist, sample)

    Returns:
        xyz (np.ndarray): Normalized point cloud (N, 3)
        original_xyz (np.ndarray): Original point cloud before normalization (N, 3)
    """
    xyz, _ = load_ply(ply_path)
    original_xyz = xyz.copy()

    N = len(xyz)
    if N != num_points:
        choice = np.random.choice(N, num_points, replace=(N < num_points))
        xyz = xyz[choice]

    xyz -= xyz.mean(0)
    xyz /= np.abs(xyz).max() + 1e-8

    return xyz, original_xyz


def predict_labels(model, xyz, device, batch_size=32):
    """
    Perform inference on point cloud.

    Args:
        model: Trained segmentation model
        xyz (np.ndarray): Point cloud (N, 3) - should be normalized
        device: torch device
        batch_size (int): Batch size for inference

    Returns:
        labels (np.ndarray): Predicted labels (N,)
        logits (np.ndarray): Raw logits (N, num_classes)
    """
    model.eval()

    xyz_tensor = torch.from_numpy(xyz).unsqueeze(0).to(device)  # (1, N, 3)

    with torch.no_grad():
        logits = model(xyz_tensor)  # (1, N, num_classes)

    logits = logits.squeeze(0).cpu().numpy()  # (N, num_classes)
    labels = logits.argmax(-1)  # (N,)

    return labels, logits


def create_inference_pipeline(
    ply_path, model_path, model_name="PointNet", device="cuda"
):
    """
    Complete inference pipeline: load model, load point cloud, predict labels.

    Args:
        ply_path (str): Path to input PLY file
        model_path (str): Path to saved model checkpoint
        model_name (str): Name of model ("PointNet", "PointNet++", or "DGCNN")
        device (str): Device to run inference on

    Returns:
        coords (np.ndarray): Point cloud coordinates (N, 3) - normalized
        labels (np.ndarray): Predicted labels (N,)
        logits (np.ndarray): Model logits (N, num_classes)
        original_coords (np.ndarray): Original coordinates before normalization
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    device = torch.device(device)

    checkpoint = torch.load(model_path, map_location=device)

    state_dict = (
        checkpoint if isinstance(checkpoint, dict) else checkpoint["model_state_dict"]
    )

    last_conv_weight = None
    for key in state_dict:
        if "seg" in key and "weight" in key:
            last_conv_weight = state_dict[key]

    if last_conv_weight is None:
        raise ValueError("Could not determine num_classes from checkpoint")

    num_classes = last_conv_weight.shape[0]
    print(f"Model: {model_name}, Num classes: {num_classes}")

    model = get_model(model_name, num_classes).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loading point cloud from: {ply_path}")
    coords, original_coords = load_and_normalize_point_cloud(ply_path)
    print(f"Point cloud shape: {coords.shape}")

    print("Running inference...")
    labels, logits = predict_labels(model, coords, device)
    print(f"Prediction complete. Unique labels: {np.unique(labels)}")

    return coords, labels, logits, original_coords


def visualize_point_cloud_matplotlib(
    coords, labels, output_path=None, title="Point Cloud Segmentation"
):
    """
    Visualize point cloud with matplotlib (static 3D plot).

    Args:
        coords (np.ndarray): Point cloud coordinates (N, 3)
        labels (np.ndarray): Labels for each point (N,)
        output_path (str): Path to save figure. If None, display instead
        title (str): Plot title

    Returns:
        fig: matplotlib figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab20", len(np.unique(labels)))

    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2], c=labels, cmap=cmap, s=10, alpha=0.6
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Label")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved matplotlib plot to: {output_path}")
    else:
        plt.show()

    return fig


def visualize_point_cloud_plotly(
    coords, labels, output_path=None, title="Point Cloud Segmentation"
):
    """
    Visualize point cloud with plotly (interactive 3D plot).

    Args:
        coords (np.ndarray): Point cloud coordinates (N, 3)
        labels (np.ndarray): Labels for each point (N,)
        output_path (str): Path to save HTML. If None, just create figure
        title (str): Plot title

    Returns:
        fig: plotly figure
    """
    unique_labels = np.unique(labels)
    color_map = {label: i for i, label in enumerate(unique_labels)}
    colors = np.array([color_map[l] for l in labels])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Label"),
                    opacity=0.8,
                ),
                text=[
                    f"Label: {l}</br>Pos: ({x:.2f}, {y:.2f}, {z:.2f})"
                    for l, x, y, z in zip(
                        labels, coords[:, 0], coords[:, 1], coords[:, 2]
                    )
                ],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        width=1000,
        height=800,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved interactive plot to: {output_path}")

    return fig


def main(ply_path, model_path, model_name="PointNet", output_dir=None, device="cuda"):
    """
    Main inference function.

    Args:
        ply_path (str): Path to input PLY file
        model_path (str): Path to saved model checkpoint
        model_name (str): Name of model architecture
        output_dir (str): Directory to save visualizations. If None, use current directory
        device (str): Device for inference ("cuda" or "cpu")
    """
    if output_dir is None:
        output_dir = os.path.dirname(ply_path)

    os.makedirs(output_dir, exist_ok=True)

    coords, labels, logits, original_coords = create_inference_pipeline(
        ply_path, model_path, model_name, device
    )

    print(f"\nResults:")
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")
    print(f"  Label distribution: {np.bincount(labels)}")

    results = {
        "coords": coords,
        "labels": labels,
        "logits": logits,
        "original_coords": original_coords,
    }

    results_path = os.path.join(output_dir, "inference_results.npz")
    np.savez(results_path, **results)
    print(f"\nSaved results to: {results_path}")

    base_name = os.path.splitext(os.path.basename(ply_path))[0]

    fig_path = os.path.join(output_dir, f"{base_name}_segmentation_matplotlib.png")
    visualize_point_cloud_matplotlib(
        coords, labels, fig_path, title=f"{model_name} - Segmentation"
    )

    html_path = os.path.join(output_dir, f"{base_name}_segmentation_interactive.html")
    visualize_point_cloud_plotly(
        coords, labels, html_path, title=f"{model_name} - Segmentation (Interactive)"
    )

    print(f"\nVisualization saved!")
    print(f"  Static plot: {fig_path}")
    print(f"  Interactive plot: {html_path}")

    return coords, labels, logits, original_coords


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Point Cloud Segmentation Inference")
    parser.add_argument("--ply", type=str, required=True, help="Path to PLY file")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DGCNN",
        choices=["PointNet", "PointNet++", "DGCNN"],
        help="Model architecture name",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Device for inference",
    )

    args = parser.parse_args()

    main(args.ply, args.model, args.model_name, args.output_dir, args.device)
