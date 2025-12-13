import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
import matplotlib.pyplot as plt


class TNet(nn.Module):
    """T-Net learns a transformation matrix for input or feature alignment"""

    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        # Shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling
        x = torch.max(x, 2, keepdim=False)[0]

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize as identity matrix
        identity = (
            torch.eye(self.k, device=x.device)
            .flatten()
            .view(1, -1)
            .repeat(batch_size, 1)
        )
        x = x + identity
        x = x.view(-1, self.k, self.k)

        return x


class PointNetFeature(nn.Module):
    """PointNet feature extractor"""

    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetFeature, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        # Input transform
        self.tnet1 = TNet(k=3)

        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transform
        if self.feature_transform:
            self.tnet2 = TNet(k=64)

        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size(2)

        # Input transform
        trans1 = self.tnet1(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)
        x = x.transpose(2, 1)

        # Shared MLP [64, 64]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        if self.feature_transform:
            trans2 = self.tnet2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # Store point features before max pooling
        point_feat = x

        # Shared MLP [64, 128, 1024]
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling for global feature
        global_feat = torch.max(x, 2, keepdim=True)[0]

        if self.global_feat:
            return global_feat, trans1, trans2
        else:
            # Expand and concatenate with point features for segmentation
            global_feat = global_feat.repeat(1, 1, n_pts)
            return torch.cat([point_feat, global_feat], 1), trans1, trans2


class PointNetSegmentation(nn.Module):
    """PointNet for semantic segmentation"""

    def __init__(self, num_classes, dropout=0.3, feature_transform=True):
        super(PointNetSegmentation, self).__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform

        # Feature extractor
        self.feat = PointNetFeature(
            global_feat=False, feature_transform=feature_transform
        )

        # Segmentation head
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [batch_size, 3, num_points]
        x, trans1, trans2 = self.feat(x)

        # Segmentation MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)

        # x: [batch_size, num_classes, num_points]
        x = x.transpose(2, 1).contiguous()
        # x: [batch_size, num_points, num_classes]

        return F.log_softmax(x, dim=-1), trans1, trans2


class PointNetSegmentationLightning(pl.LightningModule):
    """
    PyTorch Lightning Module for PointNet Segmentation
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = PointNetSegmentation(
            num_classes=cfg.data.num_classes,
            dropout=cfg.model.dropout,
            feature_transform=cfg.model.feature_transform,
        )

        if cfg.data.class_weights is not None:
            class_weights = torch.tensor(cfg.data.class_weights, dtype=torch.float).to(
                self.cfg.training.device
            )

        self.criterion = nn.NLLLoss(weight=class_weights)

        self.accuracy = Accuracy(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
        )

        self.IoU = JaccardIndex(
            task="multiclass", num_classes=cfg.data.num_classes, average="macro"
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
        # points: (batch_size, 3, num_points)
        # labels: (batch_size, num_points)

        logits, input_transform, feature_transform = self(points)
        # logits: (batch_size, num_points, num_classes)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(logits, labels)

        # Feature transform regularization
        if self.cfg.model.feature_transform and feature_transform is not None:
            reg_loss = self.feature_transform_regularizer(feature_transform)
            loss = loss + self.cfg.loss.feature_transform_reg_weight * reg_loss
            self.log("train_reg_loss", reg_loss, prog_bar=False)

        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy(preds, labels)
        iou = self.IoU(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        points, labels = batch
        logits, _, _ = self(points)
        # logits: (batch_size, num_points, num_classes)

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy(preds, labels)
        iou = self.IoU(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        points, labels = batch
        logits, _, _ = self(points)
        # logits: (batch_size, num_points, num_classes)

        # Store original shape for visualization
        original_batch_size = points.shape[0]
        original_num_points = points.shape[2]

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        acc = self.accuracy(preds, labels)
        iou = self.IoU(preds, labels)

        # Store samples for visualization
        if len(self.test_predictions) < self.cfg.evaluation.visualize_samples:
            # Reshape predictions and labels to original dimensions
            original_preds = (
                preds.view(original_batch_size, original_num_points).cpu().numpy()
            )
            original_labels = (
                labels.view(original_batch_size, original_num_points).cpu().numpy()
            )
            original_points = points.cpu().numpy()

            # Add each sample separately for correct visualization
            for i in range(
                min(
                    len(original_preds),
                    self.cfg.evaluation.visualize_samples - len(self.test_predictions),
                )
            ):
                self.test_predictions.append(original_preds[i])
                self.test_labels.append(original_labels[i])
                self.test_point_clouds.append(original_points[i])

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_iou", iou, prog_bar=True)

        return loss

    def on_test_epoch_end(self):
        """Visualize segmentation results"""
        if self.cfg.evaluation.visualize_results and len(self.test_predictions) > 0:
            self.visualize_segmentation()

    def visualize_segmentation(self):
        """Visualize point cloud segmentation results"""
        num_samples = min(
            self.cfg.evaluation.visualize_samples, len(self.test_predictions)
        )

        fig = plt.figure(figsize=(15, 4 * num_samples))

        for i in range(num_samples):
            points = self.test_point_clouds[i]  # (3, N)
            pred = self.test_predictions[i]  # (N,)
            gt = self.test_labels[i]  # (N,)

            # Transpose to (N, 3) for plotting
            points = points.transpose(1, 0)

            # Ground truth
            ax1 = fig.add_subplot(num_samples, 2, 2 * i + 1, projection="3d")
            scatter1 = ax1.scatter(
                points[:, 0], points[:, 1], points[:, 2], c=gt, cmap="tab10", s=1
            )
            ax1.set_title(f"Sample {i + 1}: Ground Truth")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            plt.colorbar(scatter1, ax=ax1, shrink=0.5)

            # Prediction
            ax2 = fig.add_subplot(num_samples, 2, 2 * i + 2, projection="3d")
            scatter2 = ax2.scatter(
                points[:, 0], points[:, 1], points[:, 2], c=pred, cmap="tab10", s=1
            )
            ax2.set_title(f"Sample {i + 1}: Prediction")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Z")
            plt.colorbar(scatter2, ax=ax2, shrink=0.5)

        plt.tight_layout()
        self.logger.experiment.add_figure(
            "segmentation_results", plt.gcf(), self.current_epoch
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
