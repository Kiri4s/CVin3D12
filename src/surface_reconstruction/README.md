# Surface Reconstruction

Adaptive surface reconstruction from segmented LiDAR point clouds (`.ply` with `scalar_Label`).

## Pipeline

```
load → preprocess → segment → analyse → classify → select method → reconstruct → assemble → evaluate
```

| Module | Responsibility |
|---|---|
| `loader.py` | Read ASCII `.ply`, extract xyz + `scalar_Label`, drop NaN |
| `preprocessing.py` | Statistical outlier removal, centred normalisation, optional voxel downsampling |
| `segmentation.py` | Group points by label, drop segments below `min_points` |
| `geometry.py` | PCA eigenvalues → linearity / planarity / sphericity; mean NN distance; normal consistency |
| `classification.py` | Rule-based type: **planar / tubular / spherical / complex** |
| `method_selector.py` | Config-driven map: geometry type → reconstruction algorithm |
| `reconstruction.py` | Poisson / Alpha Shapes / Ball Pivoting (BPA radii auto-scaled from NN distance) |
| `assembly.py` | Merge segment meshes, deduplicate vertices and triangles |
| `evaluation.py` | Point-to-mesh distance (mean / max / RMS) + triangle normal std |
| `pipeline.py` | Orchestrator for one cloud |
| `main.py` | Hydra entry point, iterates the dataset |

## Configuration

All parameters live in one file: `conf/config.yaml`.

```
conf/config.yaml
  dataset_dir          path to the folder of .ply files
  output_dir           where reconstructed meshes are written
  metrics_file         path for the JSON metrics output
  preprocessing.*      outlier removal, normalisation, voxel downsampling toggles
  segmentation.*       min_points threshold
  classification.*     planarity / linearity / sphericity thresholds
  reconstruction.*     algorithm parameters and type→method mapping
```

## Usage

```bash
# process the full dataset with default config
uv run main.py

# override any parameter from the command line (Hydra syntax)
uv run main.py \
    dataset_dir=/path/to/data \
    segmentation.min_points=50 \
    reconstruction.poisson.depth=9
```

## Output

```
outputs/
  reconstructed/
    <cloud_name>/
      segment_<label>.ply   # per-segment mesh
    <cloud_name>_combined.ply
  metrics.json              # quality metrics for every cloud and segment
```

Metrics per cloud: `point_to_mesh_mean`, `point_to_mesh_max`, `point_to_mesh_rms`, `triangle_normal_std`, vertex/triangle counts.

## Data format

Input `.ply` files must be ASCII with `scalar_Label` as an integer vertex property:

```
ply
format ascii 1.0
element vertex N
property float x
property float y
property float z
property int scalar_Label
end_header
...
```

## Results

[valve_0001_lidar_classes — interactive visualization](https://kiri4s.github.io/CVin3D12/src/surface_reconstruction/outputs/reconstructed/valve_0001_lidar_classes_visualization.html)

### valve_0001_lidar_classes — combined mesh

| Vertices | Triangles | Mean dist | Max dist | RMS dist | Normal std |
|---|---|---|---|---|---|
| 34 441 | 67 360 | 0.00283 | 0.0500 | 0.00578 | 0.576 |

### Per-segment breakdown

| Label | Geo type | Method | Vertices | Triangles | Mean dist | Max dist | RMS dist | Normal std |
|---|---|---|---|---|---|---|---|---|
| 0 | tubular | ball_pivoting | 959 | 1 806 | 0.00000 | 0.00000 | 0.00000 | 0.499 |
| 1 | tubular | ball_pivoting | 71 | 82 | 0.00109 | 0.02646 | 0.00459 | 0.577 |
| 2 | spherical | poisson | 8 528 | 16 804 | 0.00527 | 0.05362 | 0.00884 | 0.577 |
| 4 | spherical | poisson | 7 457 | 14 756 | 0.00048 | 0.01329 | 0.00086 | 0.575 |
| 5 | tubular | ball_pivoting | 430 | 616 | 0.00056 | 0.02959 | 0.00296 | 0.503 |
| 6 | complex | poisson | 7 464 | 14 756 | 0.00350 | 0.04051 | 0.00520 | 0.566 |
| 7 | complex | poisson | 3 846 | 7 446 | 0.00209 | 0.02102 | 0.00407 | 0.568 |
| 8 | complex | poisson | 5 687 | 11 096 | 0.00549 | 0.04270 | 0.00795 | 0.550 |