import random
import numpy as np
import torch

CLASSES = [
    "unknown", "pipe", "wire", "wall", "floor", "ceiling",
    "machine", "desk", "rack", "boiler", "conveyor", "structure",
    "infrastructure", "roof", "window", "door", "gate", "terrain", "facade",
]

CLASS_PALETTE = np.array([
    [128, 128, 128],  # unknown
    [255,   0,   0],  # pipe
    [255, 165,   0],  # wire
    [  0, 128,   0],  # wall
    [  0,   0, 128],  # floor
    [135, 206, 235],  # ceiling
    [255,   0, 255],  # machine
    [128,   0, 128],  # desk
    [  0, 255, 255],  # rack
    [255, 215,   0],  # boiler
    [139,  69,  19],  # conveyor
    [ 64,  64,  64],  # structure
    [  0, 255,   0],  # infrastructure
    [255, 105, 180],  # roof
    [173, 216, 230],  # window
    [210, 180, 140],  # door
    [255,  20, 147],  # gate
    [ 34, 139,  34],  # terrain
    [250, 128, 114],  # facade
], dtype=np.uint8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
