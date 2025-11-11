from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_point_cloud(path: Path, max_rows=None) -> np.ndarray:
    """Загружает .txt файл Семантик3D.

    Ожидается строка с числами, разделёнными пробелом. Возвращает numpy array
    размера (N, M).
    """

    try:
        arr = np.genfromtxt(path, max_rows=None)
    except Exception as e:
        print(f"loadtxt failed with {e}")
        exit(1)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_labels(path: Path, max_rows=None) -> np.ndarray:
    """Загружает файл .label, где каждая строка — целое число метки."""
    try:
        labels = np.genfromtxt(path, dtype=int, max_rows=max_rows)
    except Exception as e:
        print(f"loadtxt failed with {e}")
        exit(1)
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    return labels


def normalize_coordinates(
    coords: np.ndarray, method: str = "center_scale"
) -> np.ndarray:
    """Нормализует координаты.

    Поддерживаемые методы:
    - 'center_scale' : центрирование (вычитание среднего) и деление на max(abs)
    - 'minmax' : min-max масштабирование в [0,1]
    - 'zscore' : (x-mean)/std
    """
    coords = coords.astype(np.float32)
    if method == "center_scale":
        c = coords - coords.mean(axis=0)
        s = np.max(np.abs(c))
        if s == 0:
            return c
        return c / s
    elif method == "minmax":
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        return (coords - mins) / denom
    elif method == "zscore":
        mean = coords.mean(axis=0)
        std = coords.std(axis=0)
        std[std == 0] = 1.0
        return (coords - mean) / std
    else:
        raise ValueError(f"Unknown coords normalization method: {method}")


def normalize_intensity(intensity: np.ndarray, method: str = "max") -> np.ndarray:
    """Нормализует интенсивность.

    Поддерживаемые методы:
    - 'max' : деление на максимум
    - 'minmax' : масштабирование в [0,1]
    - 'zscore'
    """
    intensity = intensity.astype(np.float32)
    if method == "max":
        m = intensity.max()
        if m == 0:
            return intensity
        return intensity / m
    elif method == "minmax":
        lo = intensity.min()
        hi = intensity.max()
        denom = hi - lo
        if denom == 0:
            return intensity - lo
        return (intensity - lo) / denom
    elif method == "zscore":
        mean = intensity.mean()
        std = intensity.std()
        if std == 0:
            return intensity - mean
        return (intensity - mean) / std
    else:
        raise ValueError(f"Unknown intensity normalization method: {method}")


def visualize_labels_distribution(labels: np.ndarray, out_dir: Path) -> None:
    """Создаёт гистограмму распределения меток и сохраняет её в out_path."""
    hist_path = out_dir / "labels_hist.png"
    plt.figure()
    plt.hist(labels, bins=np.arange(labels.min(), labels.max() + 2) - 0.5)
    plt.xlabel("label")
    plt.ylabel("count")
    plt.title("Labels distribution")
    plt.tight_layout()
    plt.savefig(hist_path)
