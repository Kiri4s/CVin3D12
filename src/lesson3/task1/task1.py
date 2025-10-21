import os
from pathlib import Path
import numpy as np
import h5py
import logging

import hydra
from omegaconf import DictConfig

from utils import (
    load_point_cloud,
    load_labels,
    normalize_coordinates,
    normalize_intensity,
    ensure_output_dir,
    visualize_labels_distribution,
)

log = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="conf.yaml", version_base=None)
def main(cfg: DictConfig):
    input_txt = Path(cfg.input_txt)
    input_label = Path(cfg.input_label)
    out_dir = Path(cfg.output_dir)
    ensure_output_dir(out_dir)

    log.info(f"Loading point cloud from {input_txt}")
    pts = load_point_cloud(input_txt)
    log.info(f"Loaded point cloud shape: {pts.shape}")

    log.info(f"Loading labels from {input_label}")
    labels = load_labels(input_label)
    log.info(f"Loaded labels shape: {labels.shape}")

    if len(labels) != pts.shape[0]:
        raise ValueError(
            f"Number of points ({pts.shape[0]}) and labels ({len(labels)}) differ"
        )

    if pts.shape[1] < 7:
        if pts.shape[1] == 6:
            intensity = np.zeros((pts.shape[0], 1), dtype=pts.dtype)
            pts = np.hstack([pts, intensity])
        else:
            raise ValueError("Ожидается не менее 6 столбцов: X Y Z R G B [, intensity]")

    coords = pts[:, :3].astype(np.float32)
    rgb = pts[:, 4:].astype(np.float32)
    intensity = pts[:, 3].astype(np.float32)

    if cfg.normalize.rgb_divide_by_255:
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

    coords_norm = normalize_coordinates(coords, method=cfg.normalize.coords_method)

    if cfg.normalize.normalize_intensity:
        intensity = normalize_intensity(
            intensity, method=cfg.normalize.intensity_method
        )

    intensity = intensity.reshape(-1, 1)
    labels_col = labels.reshape(-1, 1).astype(np.int32)

    # Объединяем: X Y Z R G B intensity label
    dataset = np.hstack([coords_norm, rgb, intensity, labels_col])

    npy_path = out_dir / cfg.outputs.npy_name
    log.info(f"Saving .npy to {npy_path}")
    np.save(npy_path, dataset)

    if "txt" in cfg.outputs.formats:
        txt_path = out_dir / cfg.outputs.txt_name
        log.info(f"Saving .txt to {txt_path}")
        # Форматируем: последний столбец — метка целочисленная
        fmt = "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %d"
        np.savetxt(txt_path, dataset, fmt=fmt)

    if "h5" in cfg.outputs.formats:
        h5_path = out_dir / cfg.outputs.h5_name
        log.info(f"Saving .h5 to {h5_path}")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("dataset", data=dataset, compression="gzip")

    preview_path = out_dir / "preview_first5.txt"
    np.savetxt(preview_path, dataset[:5], fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %d")

    # Визуализация распределения меток (гистограмма)
    visualize_labels_distribution(labels, out_dir=out_dir)

    log.info(f"Dataset shape: {dataset.shape}")
    log.info(f"Saved files in {out_dir.resolve()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
