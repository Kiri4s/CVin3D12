"""
Aggregate metrics.json files produced by a Hydra multirun.

Usage:
    uv run python -m semantic_segmentation.aggregate <multirun_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np


def aggregate(multirun_dir: str) -> None:
    files = sorted(Path(multirun_dir).rglob("metrics.json"))
    if not files:
        print(f"No metrics.json found under {multirun_dir}")
        return

    print(f"Found {len(files)} run(s):")
    for f in files:
        print(f"  {f}")

    runs = [json.loads(f.read_text()) for f in files]
    scalar_keys = ["overall_accuracy", "macro_f1", "miou"]

    print("\nSummary (mean ± std):")
    for key in scalar_keys:
        vals = np.array([r[key] for r in runs])
        print(f"  {key:25s}: {vals.mean():.4f} ± {vals.std():.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    aggregate(sys.argv[1])
