import torch
import pytorch_lightning as pl
from typing import List
from model import PointNetSegmentationLightning
import os
import fire
from dataset import load_ply_file, sample_point_cloud, normalize_point_cloud


def load_ckpt(path: str) -> pl.LightningModule:
    """
    Load a PyTorch Lightning checkpoint from the specified path.

    Args:
        path (str): Path to the checkpoint file to load

    Returns:
        pl.LightningModule: Loaded model instance
    """
    print(f"Loading checkpoint from: {path}")
    model = PointNetSegmentationLightning.load_from_checkpoint(path, map_location="cpu")
    return model


def get_torch_model(ckpt: pl.LightningModule) -> torch.nn.Module:
    """
    Extract the underlying PyTorch model from a PyTorch Lightning module.

    Args:
        ckpt (pl.LightningModule): PyTorch Lightning module containing the model

    Returns:
        torch.nn.Module: The underlying neural network model
    """
    return ckpt.model


def get_device() -> torch.device:
    """
    Get the appropriate computing device (CUDA, MPS, or CPU) based on availability.

    Returns:
        torch.device: The most suitable device for computation
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def get_input_batch(inputpaths: List[str]) -> torch.Tensor:
    """
    Create a batch of input tensors from PLY file paths.

    This function loads PLY files, samples point clouds, normalizes them,
    and creates a tensor batch ready for model inference.

    Args:
        inputpaths (List[str]): List of paths to PLY files to be processed

    Returns:
        torch.Tensor: Batch of point cloud tensors with shape [batch_size, 3, num_points]
    """
    batch = []
    for file_path in inputpaths:
        try:
            vertices, labels = load_ply_file(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        vertices, _ = sample_point_cloud(vertices, labels, num_points=4096)
        point_cloud = normalize_point_cloud(vertices)
        point_cloud = torch.FloatTensor(point_cloud)
        point_cloud = point_cloud.transpose(0, 1)
        batch.append(point_cloud)

    return torch.stack(batch)


def fill_ply_with_predictions(
    inputpaths: List[str], predictions: torch.Tensor, folderpath: str
):
    """
    Save model predictions to PLY files with predicted labels.

    This function takes input PLY files and their corresponding model predictions,
    then creates new PLY files with the predicted labels included.

    Args:
        inputpaths (List[str]): List of paths to original PLY files
        predictions (torch.Tensor): Model predictions for each point cloud
        folderpath (str): Directory to save the prediction PLY files
    """
    for i, file_path in enumerate(inputpaths):
        verts, labels = load_ply_file(file_path)
        verts, _ = sample_point_cloud(verts, labels, num_points=4096)
        pred_labels = predictions[i].cpu().numpy()
        ply_lines = []
        ply_lines.append("ply\n")
        ply_lines.append("format ascii 1.0\n")
        ply_lines.append(f"element vertex {len(verts)}\n")
        ply_lines.append("property float x\n")
        ply_lines.append("property float y\n")
        ply_lines.append("property float z\n")
        ply_lines.append("property int scalar_Label\n")
        ply_lines.append("end_header\n")
        for j, vertex in enumerate(verts):
            x, y, z = vertex
            label = pred_labels[j]
            ply_lines.append(f"{x} {y} {z} {label}\n")

        pred_ply_path = f"{folderpath}/pred_{os.path.basename(file_path)}"
        with open(pred_ply_path, "w") as f:
            f.writelines(ply_lines)


def predict(
    model: torch.nn.Module, input: torch.Tensor, labels: bool = True
) -> torch.Tensor:
    """
    Perform inference using the provided model on the input data.

    This function runs the model in evaluation mode and returns predictions.
    If labels=True, it returns the argmax of the output logits.

    Args:
        model (torch.nn.Module): The trained model to use for inference
        input (torch.Tensor): Input tensor for the model (point clouds)
        labels (bool, optional): Whether to return class labels (argmax) or raw logits.
                                Defaults to True.

    Returns:
        torch.Tensor: Model predictions - either class labels or raw logits depending on
                     the 'labels' parameter
    """
    model.eval()
    with torch.no_grad():
        outputs, _, _ = model(input)

    if labels:
        outputs = torch.argmax(outputs, dim=-1)

    return outputs


def main(
    path2model: str,
    input: str | List[str],
    savepreds2folder: str = None,
    max_batch_size: int = 4,
) -> None:
    """
    Main function to run model inference on PLY files.

    This function orchestrates the entire prediction pipeline: loading the model,
    processing input files in batches, generating predictions, and optionally saving results.

    Args:
        path2model (str): Path to the model checkpoint file
        input (str | List[str]): Input PLY file(s) or directory containing PLY files
        savepreds2folder (str, optional): Directory to save prediction results.
                                         If None, results won't be saved. Defaults to None.
        max_batch_size (int, optional): Maximum number of files to process in a batch.
                                       Defaults to 4.
    """
    if isinstance(input, str):
        if os.path.isdir(input):
            input = [
                os.path.join(input, f) for f in os.listdir(input) if f.endswith(".ply")
            ]
        else:
            input = [input]

    device = get_device()
    print(f"Using device: {device}")
    model = get_torch_model(load_ckpt(path2model)).to(device)
    for bch in range(0, len(input), max_batch_size):
        input_slice = input[bch : bch + max_batch_size]
        input_batch = get_input_batch(input_slice).to(device)
        predictions = predict(model, input_batch)
        for i, file_path in enumerate(input_slice):
            print(f"Prediction for {file_path}: {predictions[i]}")

        if savepreds2folder is not None:
            os.makedirs(savepreds2folder, exist_ok=True)
            fill_ply_with_predictions(
                input_slice, predictions, folderpath=savepreds2folder
            )


if __name__ == "__main__":
    fire.Fire(main)
