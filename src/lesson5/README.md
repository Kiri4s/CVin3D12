# PointNet Model Implementation

## Overview

This lesson implements a PointNet-based deep learning model for point cloud segmentation using PyTorch Lightning.

## Project Structure

- **main.py** — Main training script with Hydra configuration management
- **model.py** — PointNet segmentation model implementation using PyTorch Lightning
- **dataset.py** — ModelNet segmentation dataset loader and PyTorch Lightning DataModule
- **inspectds.py** — Script to collect dataset statistics
- **predict.py** — Script for performing inference with trained models on PLY point cloud files
- **visualize.py** — Script for visualizing point cloud predictions and comparing with ground truth
- **conf.yaml** — Configuration file for segmentation training hyperparameters


## Dataset Format

The model expects a dataset in the following format:
- Point cloud files in `.ply` format
- Each `.ply` file should contain vertex positions (x, y, z) and labels for each point
- Labels should be integer values representing different semantic classes
- Files should be placed in the directory specified by `data.root` in the configuration

Example `.ply` file format:
```
ply
format ascii 1.0
element vertex 1000
property float x
property float y
property float z
property int label
end_header
0.1 0.2 0.3 1
0.4 0.5 0.6 2
...
```

## Usage

### Inspect Dataset Statistics

Run the inspect dataset script to collect statistics:

```bash
uv run inspectds.py
```

### Training

Run the training script with default configuration:

```bash
uv run main.py
```

Override configuration parameters from command line:

```bash
uv run main.py training.max_epochs=100 data.num_points=2048 training.batch_size=32
```

### Prediction

Perform inference on PLY files using a trained model checkpoint:

```bash
uv run predict.py --path2model=/path/to/checkpoint.ckpt --input=/path/to/input.ply
```

To process multiple files or a directory of PLY files:

```bash
uv run predict.py --path2model=/path/to/checkpoint.ckpt --input=/path/to/input/directory --savepreds2folder=/path/to/output/directory
```

### Visualization

Visualize predictions from a trained model on a PLY file:

```bash
uv run visualize.py --path2model=/path/to/checkpoint.ckpt --input=/path/to/input.ply
```

To save visualization to a file:

```bash
uv run visualize.py --path2model=/path/to/checkpoint.ckpt --input=/path/to/input.ply --save2path=/path/to/output.png
```

### Logs monitoring

```bash
uv run tensorboard --logdir=logs/pointnet
```

## Configuration

Edit `conf.yaml` to adjust:
- Data parameters (number of points, batch size, number of classes, ...)
- Training hyperparameters (learning rate, epochs, device, optimizer settings, ...)
- Model parameters (dropout, feature transforms, ...)
- Checkpoint and logging directories
- Loss function parameters (regularization weights, label smoothing, ...)

## Results

Training with configuration in `conf.yaml` file delivers results:

### Metrics

| Test metric |  DataLoader 0 |
|:-----------:|:-------------:|
| test_acc    | 0.89559841156 |
| test_iou    | 0.79310548305 |
| test_loss   | 0.25627177953 |

### Visuals

Predictions on 4 test samples:

<div style="display:flex; gap:8px; align-items:center;">
  <img src="./testresult.png" alt="1" style="width:100%; max-width:1200px;">
</div>