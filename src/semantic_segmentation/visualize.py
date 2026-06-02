from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.dataset import PointCloudDataset, load_ply
from utils import CLASS_PALETTE, CLASSES, set_seed


def _colorize(labels: np.ndarray) -> np.ndarray:
    return CLASS_PALETTE[labels % len(CLASS_PALETTE)].astype(np.float32) / 255.0


def _save_matplotlib(xyz: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                     path: Path) -> None:
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(16, 7))
    for i, (labels, title) in enumerate([(gt, "Ground Truth"), (pred, "Predicted")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                   c=_colorize(labels), s=0.5, depthshade=False)
        ax.set_title(title)
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_ply(xyz: np.ndarray, labels: np.ndarray, path: Path) -> None:
    colors = CLASS_PALETTE[labels % len(CLASS_PALETTE)]
    data = np.hstack([xyz, colors.astype(np.float32), labels[:, None]])
    with open(path, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property int label\nend_header\n"
        )
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d %d")


def _save_plotly_html(xyz_full: np.ndarray, rgb_full: np.ndarray,
                      xyz: np.ndarray, gt: np.ndarray,
                      pred: np.ndarray, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}] * 3],
        subplot_titles=["Original Colors", "Ground Truth", "Predicted"],
    )

    rgb_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in rgb_full]
    fig.add_trace(
        go.Scatter3d(
            x=xyz_full[:, 0], y=xyz_full[:, 1], z=xyz_full[:, 2],
            mode="markers",
            marker=dict(size=2, color=rgb_hex),
            name="original", showlegend=False,
        ),
        row=1, col=1,
    )

    for col, labels in enumerate([gt, pred], start=2):
        for cls_idx in np.unique(labels):
            mask = labels == cls_idx
            c = CLASS_PALETTE[cls_idx % len(CLASS_PALETTE)]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
                    mode="markers",
                    marker=dict(size=2, color=f"rgb({c[0]},{c[1]},{c[2]})"),
                    name=CLASSES[cls_idx % len(CLASSES)],
                    legendgroup=CLASSES[cls_idx % len(CLASSES)],
                    showlegend=(col == 2),
                ),
                row=1, col=col,
            )

    fig.update_layout(
        height=750,
        title_text="Semantic Segmentation",
        legend=dict(itemsizing="constant", itemwidth=40),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def _show_open3d(xyz: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    import open3d as o3d

    def make_pcd(pts, labels, offset=np.zeros(3)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts + offset)
        pcd.colors = o3d.utility.Vector3dVector(_colorize(labels))
        return pcd

    spread = xyz.max(0) - xyz.min(0)
    offset = np.array([spread[0] * 1.2, 0, 0])
    o3d.visualization.draw_geometries(
        [make_pcd(xyz, gt), make_pcd(xyz, pred, offset)],
        window_name="Ground Truth (left)  |  Predicted (right)",
    )


@hydra.main(config_path=str(Path(__file__).parent / "conf"),
            config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device, weights_only=True))
    model.eval()

    root = Path(cfg.data.root)
    test_ds = PointCloudDataset(
        [root / s for s in cfg.data.test_scenes],
        cfg.data.num_points, samples_per_cloud=1,
    )

    features, labels_gt = test_ds[cfg.viz.sample_idx]
    with torch.no_grad():
        logits = model(features.unsqueeze(0).to(device))
        preds = logits.argmax(-1).squeeze(0).cpu().numpy()

    labels_gt = labels_gt.numpy()
    xyz = features[:, :3].numpy()

    ply_path, _ = test_ds.index[cfg.viz.sample_idx]
    print(f"Visualizing: {ply_path}")
    xyz_full, rgb_full, _ = load_ply(ply_path)

    out_dir = Path(HydraConfig.get().runtime.output_dir)

    _save_matplotlib(xyz, labels_gt, preds, out_dir / "visualization.png")
    print(f"Saved visualization.png → {out_dir}")

    if cfg.viz.save_html:
        _save_plotly_html(xyz_full, rgb_full, xyz, labels_gt, preds, out_dir / "visualization.html")
        print(f"Saved visualization.html → {out_dir}")

    if cfg.viz.save_ply:
        _save_ply(xyz, labels_gt, out_dir / "gt.ply")
        _save_ply(xyz, preds, out_dir / "pred.ply")
        print(f"Saved gt.ply, pred.ply → {out_dir}")

    if cfg.viz.interactive and cfg.viz.backend == "open3d":
        _show_open3d(xyz, labels_gt, preds)


if __name__ == "__main__":
    main()
