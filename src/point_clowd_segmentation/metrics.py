import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(all_preds, all_labels, num_classes):
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    oa = (preds == labels).mean()

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    iou_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        denom = tp + fn + fp
        iou_per_class.append(tp / denom if denom > 0 else float("nan"))

    miou = np.nanmean(iou_per_class)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "OA": oa,
        "mIoU": miou,
        "IoU_per_class": iou_per_class,
        "F1": f1,
        "CM": cm,
    }
