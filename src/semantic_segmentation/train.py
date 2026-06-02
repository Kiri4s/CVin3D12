from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dataset import PointCloudDataset
from metrics import compute_metrics, save_metrics
from utils import set_seed
from tqdm.auto import tqdm


def _evaluate(model: nn.Module, loader: DataLoader,
               num_classes: int, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            all_preds.append(logits.argmax(-1).cpu().numpy().reshape(-1))
            all_labels.append(labels.numpy().reshape(-1))
    return compute_metrics(np.concatenate(all_preds),
                           np.concatenate(all_labels), num_classes)


@hydra.main(config_path=str(Path(__file__).parent / "conf"),
            config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.train.device)

    root = Path(cfg.data.root)
    train_ds = PointCloudDataset(
        [root / s for s in cfg.data.train_scenes],
        cfg.data.num_points, cfg.data.samples_per_cloud,
    )
    test_ds = PointCloudDataset(
        [root / s for s in cfg.data.test_scenes],
        cfg.data.num_points, cfg.data.samples_per_cloud,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, num_workers=cfg.train.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size,
                             shuffle=False, num_workers=cfg.train.num_workers)

    model = instantiate(cfg.model).to(device)

    class_weights = None
    if cfg.train.class_weights is not None:
        class_weights = torch.tensor(cfg.train.class_weights,
                                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    if cfg.train.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    best_miou = -1.0

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{epoch=}")
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits.reshape(-1, cfg.model.num_classes), labels.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        if epoch % 5 == 0 or epoch == cfg.train.epochs:
            m = _evaluate(model, test_loader, cfg.model.num_classes, device)
            print(f"Epoch {epoch:3d} | loss={total_loss/n:.4f} | "
                  f"OA={m['overall_accuracy']:.4f}  mIoU={m['miou']:.4f}  F1={m['macro_f1']:.4f}")
            if m["miou"] > best_miou:
                best_miou = m["miou"]
                torch.save(model.state_dict(), out_dir / "best_model.pt")

    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    final = _evaluate(model, test_loader, cfg.model.num_classes, device)
    save_metrics(final, out_dir / "metrics.json")
    print(f"\nBest model — OA={final['overall_accuracy']:.4f}  "
          f"mIoU={final['miou']:.4f}  F1={final['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
