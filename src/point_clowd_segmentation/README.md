# Point clowd segmentation

## Architectures:

- PointNet
- PointNet++
- DGCNN

## Training logs:

<div style="display:flex; gap:8px; align-items:center;">
  <img src="./results/training_curves.png" alt="DGCNN Confusion Matrix" style="width:100%; max-width:4000px;">
</div>

## Performance Metrics:

| Model | OA | mIoU | F1 |
|-------|-----|--------|---------|
| PointNet | 0.7179 | 0.3637 | 0.4581 |
| PointNet++ | 0.5232 | 0.2411 | 0.3642 |
| DGCNN | 0.8491 | 0.5835 | 0.6842 |

## Confusion Matrices:

<div style="display:flex; gap:8px; align-items:center;">
  <img src="./results/DGCNN/cm_DGCNN.png" alt="DGCNN Confusion Matrix" style="width:33%; max-width:400px;">
  <img src="./results/PointNetpp/cm_PointNetpp.png" alt="PointNet++ Confusion Matrix" style="width:33%; max-width:400px;">
  <img src="./results/PointNet/cm_PointNet.png" alt="PointNet Confusion Matrix" style="width:33%; max-width:400px;">
</div>

## Inference example:

[valve_0250_lidar_classes](https://kiri4s.github.io/CVin3D12/src/point_clowd_segmentation/results/valve_0250_lidar_classes_segmentation_interactive.html)

<div style="display:flex; gap:8px; align-items:center;">
  <img src="./results/valve_0250_lidar_classes_segmentation_matplotlib.png" alt="" style="width:100%; max-width:600px;">
</div>