"""Generate confusion_matrix.png from a metrics.json produced by train.py."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = [
    "unknown", "pipe", "wire", "wall", "floor", "ceiling",
    "machine", "desk", "rack", "boiler", "conveyor", "structure",
    "infrastructure", "roof", "window", "door", "gate", "terrain", "facade",
]


def plot(metrics_path: Path, out_path: Path) -> None:
    with open(metrics_path) as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"], dtype=np.float64)

    # keep only classes that appear in GT or predictions
    row_sums = cm.sum(1)
    col_sums = cm.sum(0)
    active = np.where((row_sums > 0) | (col_sums > 0))[0]
    cm = cm[np.ix_(active, active)]
    labels = [CLASSES[i] for i in active]

    # row-normalise (recall per class); avoid div-by-zero for pred-only classes
    row_sums = cm.sum(1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.7), max(6, n * 0.65)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=12)

    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    metrics = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).parent / "outputs/2026-06-01/first/metrics.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).parent / "outputs/conf_matrix.png"
    plot(metrics, out)
