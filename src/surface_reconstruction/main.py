"""Entry point: Hydra-managed, processes the full dataset."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from loader import load_dataset
from pipeline import run_cloud

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clouds = load_dataset(cfg.dataset_dir)
    log.info("Loaded %d clouds from %s", len(clouds), cfg.dataset_dir)

    all_metrics = []
    for cloud in clouds[:cfg.clouds_limit]:
        result = run_cloud(cloud, cfg, out_dir)
        all_metrics.append(result)

    metrics_path = Path(cfg.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    main()
