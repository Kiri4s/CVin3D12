from dataset import ValveSegmentationDataset
from torch.utils.data import DataLoader
import numpy as np


def main():
    dataset = ValveSegmentationDataset(
        root="../../../datasets/ds4segmentation",
        num_points=None,
        split="full",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Number ofsamples in the dataset: {len(dataset)}")

    points, labels = dataset[0]
    print(f"Points shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")

    labels_statistics = {}
    num_classes = 0
    for _, labels in dataloader:
        for label in labels.squeeze(0).numpy():
            labels_statistics[label] = labels_statistics.get(label, 0) + 1
    print("Labels statistics in the dataset:")
    for label, count in labels_statistics.items():
        print(f"Label {label}: {count} points")
        num_classes += 1
    weights = np.array(
        [1.0 / labels_statistics[i] for i in sorted(labels_statistics.keys())]
    )
    weights = weights / np.sum(weights)

    print(f"Number of classes: {num_classes}")
    print(f"Class weights (by frequency): {weights}")
    print(
        f"average num points per cloud: {sum(labels_statistics.values()) / len(dataloader)}"
    )


if __name__ == "__main__":
    main()
