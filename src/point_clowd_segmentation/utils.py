import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def plot_history(history, name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label=f"{name} train")
    axes[0].plot(history["val_loss"], label=f"{name} val", linestyle="--")
    axes[1].plot(history["val_miou"], label=name)
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlabel("Epoch")
    axes[1].set_title("Val mIoU")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_confusion(cm, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(max(6, cm.shape[0]), max(5, cm.shape[0] - 1)))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"cm_{model_name.replace('+', 'p')}.png"), dpi=150
    )
    plt.close()


def print_table(met, name, num_classes):
    header = f"{'Model':<14} {'OA':>7} {'mIoU':>7} {'F1':>7}"
    print("\n" + "=" * 50)
    print("MODEL COMPARISON TABLE")
    print("=" * 50)
    print(header)
    print("-" * 50)
    print(f"{name:<14} {met['OA']:>7.4f} {met['mIoU']:>7.4f} {met['F1']:>7.4f}")
    print("=" * 50)

    # per-class IoU
    print("\nPer-class IoU:")
    header2 = f"{'Model':<14}" + "".join(f"  cls{c}" for c in range(num_classes))
    print(header2)
    row = f"{name:<14}" + "".join(
        f"  {v:.2f}" if not np.isnan(v) else "   nan" for v in met["IoU_per_class"]
    )
    print(row)


def save_history(history, name, out_dir):
    with open(
        os.path.join(out_dir, f"history_{name.replace('+', 'p')}.json"), "w"
    ) as f:
        json.dump(convert_to_serializable(history), f)

def save_metrics(metrics, name, out_dir):
    with open(
        os.path.join(out_dir, f"metrics_{name.replace('+', 'p')}.json"), "w"
    ) as f:
        json.dump(convert_to_serializable(metrics), f)

def collect_results(cfg):
    results = {}
    for model_name in cfg.models:
        save_path = os.path.join(cfg.out_dir, model_name.replace("+", "p"))
        with open(
            os.path.join(save_path, f"history_{model_name.replace('+', 'p')}.json"), "r"
        ) as f:
            history = json.load(f)
        with open(
            os.path.join(save_path, f"metrics_{model_name.replace('+', 'p')}.json"), "r"
        ) as f:
            metrics = json.load(f)
        results[model_name] = {"history": history, "metrics": metrics}
    metrics_table = {}
    histories = {}
    for model_name, data in results.items():
        metrics_table[model_name] = {
            "OA": data["metrics"]["OA"],
            "mIoU": data["metrics"]["mIoU"],
            "F1": data["metrics"]["F1"],
        }
        histories[model_name] = data["history"]

    df = pd.DataFrame.from_dict(metrics_table, orient="index")
    df.to_csv(os.path.join(cfg.out_dir, "final_results.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        axes[0].plot(history["train_loss"], label=f"{name} train")
        axes[0].plot(history["val_loss"], label=f"{name} val", linestyle="--")
        axes[1].plot(history["val_miou"], label=name)
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlabel("Epoch")
    axes[1].set_title("Val mIoU")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "training_curves.png"), dpi=150)
    plt.close()
