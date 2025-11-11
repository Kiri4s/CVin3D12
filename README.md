# CVin3D Point Cloud Processing

This repository contains scripts and reports for 3D point cloud segmentation and subsampling tasks.

## Structure

- lesson1

    - [task1](./src/lesson1/task1/)

    - [task1b](./src/lesson1/task1b/)

    - [task2](./src/lesson1/task2/)

    - [task2b](./src/lesson1/task2b/)

- lesson2

    - [task1](./src/lesson2/task1/)

    - [task2](./src/lesson2/task2/)

    - [task3](./src/lesson2/task3/)

    - [task4](./src/lesson2/task4/)

- lesson3

    - [task1](./src/lesson3/task1/)

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