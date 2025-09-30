# CVin3D Point Cloud Processing

This repository contains scripts and reports for 3D point cloud segmentation and subsampling tasks.

## Structure

- [task1](./src/2025.09.30/task1/)

- [task1b](./src/2025.09.30/task1b/)

- [task2](./src/2025.09.30/task2/)

- [task2b](./src/2025.09.30/task2b/)

## Installation

1) install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2) run:

```sh
uv sync
```

## Usage

Example for task2b:

1) set the configuration in config.yaml

2) run code:

```sh
cd src/2025.09.30/task2b/ && uv run task2b.py
```

## Requirements

- Python 3.13+
- numpy
- matplotlib
- hydra-core

## Reports

See the `README.md` files in each task folder for detailed results, figures, and analysis.