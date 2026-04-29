from omegaconf import DictConfig, OmegaConf
import os
import hydra
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")

from dataset import ValveDataset, split_files, load_ply
from models import get_model
from metrics import compute_metrics
from utils import (
    print_table,
    save_history,
    save_metrics,
    collect_results,
    plot_confusion,
    plot_history,
)


def get_num_classes(files, sample=20):
    max_label = 0
    for f in files[:sample]:
        _, labels = load_ply(f)
        max_label = max(max_label, labels.max())
    return int(max_label) + 1


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for xyz, labels in loader:
        xyz, labels = xyz.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(xyz)  # B, N, C
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for xyz, labels in loader:
        xyz, labels = xyz.to(device), labels.to(device)
        logits = model(xyz)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item()
        preds = logits.argmax(-1).cpu().numpy().reshape(-1)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy().reshape(-1))
    metrics = compute_metrics(all_preds, all_labels, num_classes)
    return total_loss / len(loader), metrics


def train_model(name, train_files, val_files, test_files, num_classes, cfg, device):
    print(f"\n{'=' * 50}\nTraining {name}\n{'=' * 50}")
    train_ds = ValveDataset(train_files, cfg.num_points, augment=True)
    val_ds = ValveDataset(val_files, cfg.num_points)
    test_ds = ValveDataset(test_files, cfg.num_points)

    train_ld = DataLoader(
        train_ds,
        cfg.batch_size,
        shuffle=True,
        num_workers=4,  # , pin_memory=True
    )
    val_ld = DataLoader(
        val_ds,
        cfg.batch_size,
        shuffle=False,
        num_workers=4,  # , pin_memory=True
    )
    test_ld = DataLoader(
        test_ds,
        cfg.batch_size,
        shuffle=False,
        num_workers=4,  # , pin_memory=True
    )

    model = get_model(name, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    best_miou, history = 0, {"train_loss": [], "val_loss": [], "val_miou": []}
    ckpt_path = os.path.join(
        cfg.out_dir, name.replace("+", "p"), f"{name.replace('+', 'p')}_best.pt"
    )
    pbar = tqdm(range(1, cfg.epochs + 1), desc=f"Training {name}")
    for epoch in pbar:
        t_loss = train_epoch(model, train_ld, optimizer, criterion, device)
        v_loss, v_met = eval_epoch(model, val_ld, criterion, device, num_classes)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_miou"].append(v_met["mIoU"])

        if v_met["mIoU"] > best_miou:
            best_miou = v_met["mIoU"]
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} "
                f"| val_mIoU={v_met['mIoU']:.4f} | val_OA={v_met['OA']:.4f}"
            )
        pbar.set_postfix(
            {"train_loss": f"{t_loss:.4f}", "val_mIoU": f"{v_met['mIoU']:.4f}"}
        )

    # test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_met = eval_epoch(model, test_ld, criterion, device, num_classes)
    print(
        f"\n  Test OA={test_met['OA']:.4f}  mIoU={test_met['mIoU']:.4f}  F1={test_met['F1']:.4f}"
    )

    return history, test_met


@hydra.main(config_path="config", config_name="basic", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_files, val_files, test_files = split_files(cfg.data_dir)
    print(
        f"Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test"
    )

    num_classes = get_num_classes(train_files + val_files + test_files)
    print(f"Num classes: {num_classes}")

    for model_name in cfg.models:
        os.makedirs(
            os.path.join(cfg.out_dir, model_name.replace("+", "p")), exist_ok=True
        )
        save_path = os.path.join(cfg.out_dir, model_name.replace("+", "p"))
        h, met = train_model(
            model_name, train_files, val_files, test_files, num_classes, cfg, device
        )
        save_history(h, model_name, save_path)
        save_metrics(met, model_name, save_path)
        plot_confusion(met["CM"], model_name, save_path)
        plot_history(h, model_name, save_path)
        print_table(met, model_name, num_classes)

    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
            
    collect_results(cfg)


if __name__ == "__main__":
    main()
