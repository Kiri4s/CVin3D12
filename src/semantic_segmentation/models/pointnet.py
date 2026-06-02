import torch
import torch.nn as nn


class PointNetSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(64 + 1024, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C) -> (B, N, num_classes)"""
        x = x.transpose(2, 1)                                          # (B, C, N)
        local_feat = self.local_mlp(x)                                 # (B, 64, N)
        global_feat = self.global_mlp(local_feat).max(dim=-1)[0]      # (B, 1024)
        global_feat = global_feat.unsqueeze(-1).expand(-1, -1, local_feat.shape[-1])  # (B, 1024, N)
        out = self.head(torch.cat([local_feat, global_feat], dim=1))   # (B, num_classes, N)
        return out.transpose(2, 1)                                      # (B, N, num_classes)
