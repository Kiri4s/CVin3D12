import torch
import torch.nn as nn


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, C, N)
    returns idx: (B, N, k) — indices of k nearest neighbors (self excluded)
    """
    B, C, N = x.shape
    inner = torch.bmm(x.transpose(2, 1), x)          # (B, N, N)
    xx = (x ** 2).sum(dim=1, keepdim=True)            # (B, 1, N)
    dist = xx.transpose(2, 1) + xx - 2.0 * inner      # (B, N, N)  ||xi - xj||^2
    return dist.topk(k + 1, dim=-1, largest=False)[1][:, :, 1:]  # (B, N, k)


def _edge_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, C, N)
    idx: (B, N, k)
    returns (B, 2C, N, k)  — cat(x_neighbor - x_center, x_center)
    """
    B, C, N = x.shape
    k = idx.shape[-1]

    offset = torch.arange(B, device=x.device).view(-1, 1, 1) * N   # (B, 1, 1)
    flat = (idx + offset).reshape(-1)                                # (B*N*k,)

    x_t = x.transpose(2, 1).reshape(B * N, C)                       # (B*N, C)
    neighbors = x_t[flat].reshape(B, N, k, C)                       # (B, N, k, C)
    center = x.transpose(2, 1).unsqueeze(2).expand(-1, -1, k, -1)   # (B, N, k, C)

    edge = torch.cat([neighbors - center, center], dim=-1)           # (B, N, k, 2C)
    return edge.permute(0, 3, 1, 2)                                  # (B, 2C, N, k)


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = _knn(x, self.k)
        return self.net(_edge_feature(x, idx)).max(dim=-1)[0]  # (B, out, N)


class DGCNNSeg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, k: int,
                 emb_dims: int, dropout: float):
        super().__init__()
        self.ec1 = EdgeConv(in_channels, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 64, k)
        self.ec4 = EdgeConv(64, 128, k)

        # local concat dim: 64+64+64+128 = 320
        self.agg = nn.Sequential(
            nn.Conv1d(320, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv1d(320 + emb_dims, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C) -> (B, N, num_classes)"""
        x = x.transpose(2, 1)              # (B, C, N)

        x1 = self.ec1(x)                   # (B, 64,  N)
        x2 = self.ec2(x1)                  # (B, 64,  N)
        x3 = self.ec3(x2)                  # (B, 64,  N)
        x4 = self.ec4(x3)                  # (B, 128, N)

        local = torch.cat([x1, x2, x3, x4], dim=1)                           # (B, 320, N)
        global_feat = self.agg(local).max(dim=-1)[0]                          # (B, emb_dims)
        global_feat = global_feat.unsqueeze(-1).expand(-1, -1, local.shape[-1])  # (B, emb_dims, N)

        out = self.head(torch.cat([local, global_feat], dim=1))               # (B, num_classes, N)
        return out.transpose(2, 1)                                             # (B, N, num_classes)
