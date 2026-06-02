import json

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    oa = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro",
                               zero_division=0, labels=range(num_classes)))

    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    intersection = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - intersection
    iou = np.where(union > 0, intersection / union, np.nan)
    miou = float(np.nanmean(iou))

    return {
        "overall_accuracy": oa,
        "macro_f1": macro_f1,
        "miou": miou,
        "iou_per_class": iou.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
