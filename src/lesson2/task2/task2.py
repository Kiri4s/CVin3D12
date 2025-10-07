from dataclasses import dataclass
import os
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from matplotlib import cm

import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hu


@dataclass
class Config:
    # Генерация точек
    N: int = 1000
    seed: int = 42

    # Сохранение/папка результатов (относительно оригинальной cwd)
    save_dir: str = "./outputs"

    # Визуализация
    plot_matplotlib: bool = True
    plot_plotly: bool = True

    # Параметры маркеров
    matplotlib_marker_size: int = 10
    plotly_marker_size: int = 3

    # Сохранение файлов данных
    save_csv: bool = True
    save_ply: bool = False

    # Альтернативная цветовая схема
    use_alt_colormap: bool = True
    alt_colormap: str = "viridis"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pointcloud_csv(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Сохраняет x,y,z,r,g,b в CSV (r,g,b в диапазоне [0,1])"""
    header = "x,y,z,r,g,b"
    data = np.hstack((points, colors))
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_pointcloud_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Сохраняет простейший ASCII .ply с цветами в 0..255"""
    n = points.shape[0]
    colors255 = np.clip((colors * 255).astype(int), 0, 255)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors255):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


@hydra.main(config_path="./", config_name="config.yaml", version_base=None)
def main(cfg: Optional[DictConfig]) -> None:
    print("Config:")
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(int(cfg.seed))

    N = int(cfg.N)
    points = np.random.rand(N, 3)

    colors = points.copy()

    alt_colors = None
    if cfg.use_alt_colormap:
        center = np.array([0.5, 0.5, 0.5])
        dist = np.linalg.norm(points - center, axis=1)
        norm_dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-12)
        cmap = cm.get_cmap(cfg.alt_colormap)
        alt_colors = cmap(norm_dist)[:, :3]

    orig_cwd = hu.get_original_cwd()
    save_dir = os.path.abspath(os.path.join(orig_cwd, cfg.save_dir))
    ensure_dir(save_dir)
    print(f"Результаты будут сохранены в: {save_dir}")

    if cfg.save_csv:
        csv_path = os.path.join(save_dir, "points_rgb.csv")
        save_pointcloud_csv(csv_path, points, colors)
        print(f"Сохранено CSV: {csv_path}")

    if cfg.save_ply:
        ply_path = os.path.join(save_dir, "points_rgb.ply")
        save_pointcloud_ply(ply_path, points, colors)
        print(f"Сохранено PLY: {ply_path}")

    if cfg.plot_matplotlib:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=cfg.matplotlib_marker_size)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D визуализация облака точек (Matplotlib)")
        ax.grid(True)

        out_png = os.path.join(save_dir, "points_2d_matplotlib.png")
        out_svg = os.path.join(save_dir, "points_2d_matplotlib.svg")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        fig.savefig(out_svg)
        plt.close(fig)
        print(f"Сохранено Matplotlib (PNG): {out_png}")
        print(f"Сохранено Matplotlib (SVG): {out_svg}")

        if alt_colors is not None:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.scatter(
                points[:, 0], points[:, 1], c=alt_colors, s=cfg.matplotlib_marker_size
            )
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_title(f"2D визуализация — {cfg.alt_colormap} (по расстоянию)")
            ax2.grid(True)
            out_png2 = os.path.join(save_dir, "points_2d_alt_matplotlib.png")
            fig2.tight_layout()
            fig2.savefig(out_png2, dpi=200)
            plt.close(fig2)
            print(f"Сохранено Matplotlib (альт, PNG): {out_png2}")

    if cfg.plot_plotly:
        rgb_strings = [
            f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})" for r, g, b in colors
        ]
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=cfg.plotly_marker_size, color=rgb_strings),
                )
            ]
        )
        fig.update_layout(
            title="3D визуализация облака точек (Plotly)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        )

        out_html = os.path.join(save_dir, "points_3d_plotly.html")
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"Сохранено Plotly (HTML): {out_html}")

        if alt_colors is not None:
            rgb_strings_alt = [
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                for r, g, b in alt_colors
            ]
            fig_alt = go.Figure(
                data=[
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode="markers",
                        marker=dict(size=cfg.plotly_marker_size, color=rgb_strings_alt),
                    )
                ]
            )
            fig_alt.update_layout(
                title=f"3D (альт) — {cfg.alt_colormap}",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            )
            out_html_alt = os.path.join(save_dir, "points_3d_plotly_alt.html")
            fig_alt.write_html(out_html_alt, include_plotlyjs="cdn")
            print(f"Сохранено Plotly (альт HTML): {out_html_alt}")


if __name__ == "__main__":
    main()
