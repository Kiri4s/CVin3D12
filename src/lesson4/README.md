# PointNet Model Implementation

## Overview

This lesson implements a PointNet-based deep learning model for point cloud classification using PyTorch Lightning.

## Project Structure

- **main.py** — Main training script with Hydra configuration management

- **model.py** — PointNet model implementation using PyTorch Lightning

- **dataset.py** — ModelNet dataset loader and PyTorch Lightning DataModule

- **visualize.py** — Visualization utilities for point clouds and model outputs

- **conf.yaml** — Configuration file for training hyperparameters

## Usage

Run the training script with default configuration:

```sh
uv run main.py
```

Override configuration parameters from command line:

```sh
uv run main.py training.max_epochs=100 data.num_points=2048 training.batch_size=32
```

### Visualization

Visualize results and point clouds:

```sh
uv run visualize.py
```

## Configuration

Edit `conf.yaml` to adjust:
- Data parameters (number of points, classes, batch size)
- Model architecture settings
- Training hyperparameters (learning rate, epochs, device)
- Checkpoint and logging directories