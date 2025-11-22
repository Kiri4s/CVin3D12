import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns


class TNet(nn.Module):
    """
    Transformation Network (T-Net)
    Learns a transformation matrix for input or feature alignment
    """

    def __init__(self, k=3, mlp_dims=[64, 128, 1024]):
        super().__init__()
        self.k = k

        # Shared MLP layers
        self.conv1 = nn.Conv1d(k, mlp_dims[0], 1)
        self.conv2 = nn.Conv1d(mlp_dims[0], mlp_dims[1], 1)
        self.conv3 = nn.Conv1d(mlp_dims[1], mlp_dims[2], 1)

        self.bn1 = nn.BatchNorm1d(mlp_dims[0])
        self.bn2 = nn.BatchNorm1d(mlp_dims[1])
        self.bn3 = nn.BatchNorm1d(mlp_dims[2])

        # FC layers
        self.fc1 = nn.Linear(mlp_dims[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x):
        batch_size = x.size(0)

        # MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling over points
        x = torch.max(x, 2)[0]

        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        transform = x.view(batch_size, self.k, self.k)

        return transform


class PointNetBackbone(nn.Module):
    """
    PointNet Feature Extraction Backbone
    """

    def __init__(self, point_mlp=[64, 128, 1024], feature_transform=True):
        super().__init__()
        self.feature_transform = feature_transform

        # Input transformation
        self.input_transform = TNet(k=3)

        # MLP 1
        self.conv1 = nn.Conv1d(3, point_mlp[0], 1)
        self.conv2 = nn.Conv1d(point_mlp[0], point_mlp[1], 1)

        self.bn1 = nn.BatchNorm1d(point_mlp[0])
        self.bn2 = nn.BatchNorm1d(point_mlp[1])

        # Feature transformation
        if self.feature_transform:
            self.feature_transform_net = TNet(k=point_mlp[1])

        # MLP 2
        self.conv3 = nn.Conv1d(point_mlp[1], point_mlp[2], 1)
        self.bn3 = nn.BatchNorm1d(point_mlp[2])

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()

        # Input transformation
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)

        # MLP 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        feature_transform = None
        if self.feature_transform:
            feature_transform = self.feature_transform_net(x)
            x = torch.bmm(feature_transform, x)

        # MLP 2
        x = F.relu(self.bn3(self.conv3(x)))

        # Global feature: max pooling
        global_feature = torch.max(x, 2)[0]

        return global_feature, feature_transform


class PointNetClassifier(nn.Module):
    """
    PointNet Classification Head
    """

    def __init__(
        self,
        num_classes,
        global_feature_dim=1024,
        classifier_dims=[512, 256],
        dropout=0.3,
        feature_transform=True,
    ):
        super().__init__()

        # Backbone
        self.backbone = PointNetBackbone(
            point_mlp=[64, 128, global_feature_dim], feature_transform=feature_transform
        )

        # Classification head
        self.fc1 = nn.Linear(global_feature_dim, classifier_dims[0])
        self.fc2 = nn.Linear(classifier_dims[0], classifier_dims[1])
        self.fc3 = nn.Linear(classifier_dims[1], num_classes)

        self.bn1 = nn.BatchNorm1d(classifier_dims[0])
        self.bn2 = nn.BatchNorm1d(classifier_dims[1])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Extract features
        global_feature, feature_transform = self.backbone(x)

        # Classification
        x = F.relu(self.bn1(self.fc1(global_feature)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x, feature_transform


class PointNetLightning(pl.LightningModule):
    """
    PyTorch Lightning Module for PointNet
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = PointNetClassifier(
            num_classes=cfg.data.num_classes,
            global_feature_dim=cfg.model.global_feature_dim,
            classifier_dims=cfg.model.classifier_dims,
            dropout=cfg.model.dropout,
            feature_transform=cfg.model.feature_transform,
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

        self.train_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=cfg.data.num_classes)

        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=cfg.data.num_classes
        )

        self.test_predictions = []
        self.test_labels = []
        self.test_point_clouds = []

    def forward(self, x):
        return self.model(x)

    def feature_transform_regularizer(self, feature_transform):
        """
        Regularization for feature transformation matrix
        Encourages it to be close to orthogonal
        """
        batch_size = feature_transform.size(0)
        k = feature_transform.size(1)

        identity = torch.eye(k).unsqueeze(0).repeat(batch_size, 1, 1)
        identity = identity.to(feature_transform.device)

        mat_diff = (
            torch.bmm(feature_transform, feature_transform.transpose(1, 2)) - identity
        )
        mat_diff_loss = torch.mean(torch.norm(mat_diff, dim=(1, 2)))

        return mat_diff_loss

    def training_step(self, batch, batch_idx):
        points, labels = batch
        logits, feature_transform = self(points)
        loss = self.criterion(logits, labels)

        # Feature transform regularization
        if self.cfg.model.feature_transform and feature_transform is not None:
            reg_loss = self.feature_transform_regularizer(feature_transform)
            loss = loss + self.cfg.loss.feature_transform_reg_weight * reg_loss
            self.log("train_reg_loss", reg_loss, prog_bar=False)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        points, labels = batch
        logits, feature_transform = self(points)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        points, labels = batch
        logits, _ = self(points)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)

        if len(self.test_predictions) < self.cfg.evaluation.visualize_samples:
            self.test_predictions.extend(preds.cpu().numpy())
            self.test_labels.extend(labels.cpu().numpy())
            self.test_point_clouds.extend(points.cpu().numpy())

        self.confusion_matrix.update(preds, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        """Plot confusion matrix"""
        if self.cfg.evaluation.save_confusion_matrix:
            cm = self.confusion_matrix.compute().cpu().numpy()
            self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        """Plot and log confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        self.logger.experiment.add_figure(
            "confusion_matrix", plt.gcf(), self.current_epoch
        )
        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.optimizer.lr,
            weight_decay=self.cfg.training.optimizer.weight_decay,
        )

        if self.cfg.training.scheduler.name == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.cfg.training.scheduler.step_size,
                gamma=self.cfg.training.scheduler.gamma,
            )
            return [optimizer], [scheduler]

        return optimizer


if __name__ == "__main__":
    model = PointNetClassifier(num_classes=10)
    x = torch.randn(4, 3, 1024)
    logits, transform = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Transform shape: {transform.shape if transform is not None else None}")
