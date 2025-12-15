import torch
import matplotlib.pyplot as plt
import fire
from predict import get_torch_model, load_ckpt, get_device, load_ply_file, predict
from dataset import sample_point_cloud, normalize_point_cloud


def visualize_point_cloud(
    points: torch.Tensor,
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor = None,
    save2path: str = None,
):
    """
    Visualizes predicted and true labels for a 3D point cloud.

    Args:
        points (torch.Tensor): Tensor of 3D coordinates with shape (N, 3) where N is the number of points.
        pred_labels (torch.Tensor): Predicted labels for each point with shape (N,).
        true_labels (torch.Tensor, optional): True labels for each point with shape (N,). If provided,
                                           a side-by-side comparison will be created showing both
                                           predicted and true labels. Defaults to None.
        save2path (str, optional): Path to save the visualization. If None, the plot will be displayed
                                 on screen. Defaults to None.

    Returns:
        None: Either displays the plot or saves it to the specified path.
    """
    points = points.cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("Predicted Labels")

    labels = pred_labels.cpu().numpy()
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="jet", s=1
    )
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if true_labels is not None and len(true_labels) > 0:
        ax = fig.add_subplot(122, projection="3d")
        ax.set_title("True Labels")

        labels = true_labels.cpu().numpy()
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=labels, cmap="jet", s=1
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    if save2path:
        plt.savefig(save2path)
    else:
        plt.show()


def main(path2model: str, input: str, save2path: str = None):
    """
    Main function to load a trained model, perform prediction on a point cloud, and visualize the results.

    This function loads a model checkpoint, processes an input point cloud file, makes predictions
    using the model, and visualizes both the predicted and true labels.

    Args:
        path2model (str): Path to the model checkpoint file.
        input (str): Path to the input point cloud file (PLY format).
        save2path (str, optional): Path to save the visualization. If None, the plot will be displayed
                                 on screen. Defaults to None.

    Returns:
        None: Either displays the plot or saves it to the specified path after visualizing
              predicted vs true labels for the 3D point cloud.
    """
    device = get_device()
    print(f"Using device: {device}")

    model = get_torch_model(load_ckpt(path2model)).to(device)
    verts, labels = load_ply_file(input)
    point_cloud, labels = sample_point_cloud(verts, labels, num_points=None)
    point_cloud = normalize_point_cloud(point_cloud)
    point_cloud = torch.FloatTensor(point_cloud).to(device)
    labels = torch.LongTensor(labels)
    point_cloud = point_cloud.transpose(0, 1).unsqueeze(0)

    visualize_point_cloud(
        point_cloud, predict(model, point_cloud), labels, save2path=save2path
    )


if __name__ == "__main__":
    fire.Fire(main)
