import torch
import torch.nn as nn


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.conv(x).max(-1)[0]
        x = self.fc(x).view(B, self.k, self.k)
        return x + torch.eye(self.k, device=x.device).unsqueeze(0)


class PointNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.tnet3 = TNet(3)
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.tnet64 = TNet(64)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.seg = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x):  # x: B,N,3
        x = x.transpose(2, 1)  # B,3,N
        T3 = self.tnet3(x)
        x = torch.bmm(T3, x)
        x = self.conv1(x)
        x = self.conv2(x)
        local_feat = x  # B,64,N
        T64 = self.tnet64(x)
        x = torch.bmm(T64, x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        global_feat = x.max(-1)[0].unsqueeze(-1).expand(-1, -1, local_feat.size(-1))
        x = torch.cat([local_feat, global_feat], dim=1)  # B,1088,N
        return self.seg(x).transpose(2, 1)  # B,N,C


def fps(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[torch.arange(B), farthest].unsqueeze(1)
        d = ((xyz - c) ** 2).sum(-1)
        dist = torch.min(dist, d)
        farthest = dist.argmax(-1)
    return centroids


def ball_query(radius, nsample, xyz, new_xyz):
    dist = torch.cdist(new_xyz, xyz)  # B,S,N
    idx = dist.argsort(-1)[:, :, :nsample]
    mask = dist.gather(-1, idx) > radius
    idx[mask] = idx[:, :, :1].expand_as(idx)[mask]
    return idx  # B,S,nsample


def sample_and_group(npoint, radius, nsample, xyz, feat):
    B, N, _ = xyz.shape
    s_idx = fps(xyz, npoint)
    new_xyz = xyz[torch.arange(B).unsqueeze(-1), s_idx]  # B,S,3
    idx = ball_query(radius, nsample, xyz, new_xyz)  # B,S,K
    B, S, K = idx.shape
    grouped_xyz = xyz[
        torch.arange(B).view(B, 1, 1).expand(B, S, K), idx
    ] - new_xyz.unsqueeze(2)  # B,S,K,3
    if feat is not None:
        grouped_feat = feat[torch.arange(B).view(B, 1, 1).expand(B, S, K), idx]
        grouped = torch.cat([grouped_xyz, grouped_feat], -1)
    else:
        grouped = grouped_xyz
    return new_xyz, grouped  # B,S,3  |  B,S,K,C


class SALayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, c = [], in_ch + 3
        for out in mlp:
            layers += [nn.Conv2d(c, out, 1), nn.BatchNorm2d(out), nn.ReLU()]
            c = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feat):
        new_xyz, grouped = sample_and_group(
            self.npoint, self.radius, self.nsample, xyz, feat
        )
        x = grouped.permute(0, 3, 2, 1)  # B,C,K,S
        x = self.mlp(x).max(2)[0]  # B,C,S
        return new_xyz, x.transpose(2, 1)  # B,S,C


class FPLayer(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        layers, c = [], in_ch
        for out in mlp:
            layers += [nn.Conv1d(c, out, 1), nn.BatchNorm1d(out), nn.ReLU()]
            c = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, feat1, feat2):
        # interpolate feat2 -> xyz1
        dist = torch.cdist(xyz1, xyz2)  # B,N1,N2
        k = min(3, xyz2.shape[1])  # adaptive k based on available points
        dist, idx = dist.topk(k, largest=False)  # B,N1,k
        dist_recip = 1.0 / (dist + 1e-8)
        weight = dist_recip / dist_recip.sum(-1, keepdim=True)
        B, N1, _ = xyz1.shape
        interp = (
            feat2[torch.arange(B).view(B, 1, 1).expand(B, N1, k), idx]
            * weight.unsqueeze(-1)
        ).sum(2)  # B,N1,C
        if feat1 is not None:
            interp = torch.cat([feat1, interp], -1)
        return self.mlp(interp.transpose(2, 1)).transpose(2, 1)


class PointNetPP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.sa1 = SALayer(512, 0.2, 32, 0, [64, 64, 128])
        self.sa2 = SALayer(128, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SALayer(None, None, None, 256, [256, 512, 1024])  # global
        self.fp3 = FPLayer(1024 + 256, [256, 256])
        self.fp2 = FPLayer(256 + 128, [256, 128])
        self.fp1 = FPLayer(128 + 0, [128, 128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def _global_sa(self, xyz, feat):
        # treat all points as one group
        B, N, _ = xyz.shape
        grouped = torch.cat([xyz, feat], -1)  # B,N,C+3
        x = grouped.permute(0, 2, 1).unsqueeze(-1)  # B,C,N,1
        layers = nn.Sequential(
            nn.Conv2d(feat.size(-1) + 3, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        ).to(xyz.device)
        x = layers(x).squeeze(-1).max(-1)[0]  # B,1024
        return torch.zeros(B, 1, 3, device=xyz.device), x.unsqueeze(1)

    def forward(self, x):  # x: B,N,3
        xyz0 = x
        xyz1, f1 = self.sa1(xyz0, None)
        xyz2, f2 = self.sa2(xyz1, f1)
        # global
        B = x.size(0)
        g_in = torch.cat([xyz2, f2], -1).permute(0, 2, 1)  # B,C,S
        g_mlp = nn.Sequential(
            nn.Conv1d(f2.size(-1) + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        ).to(x.device)
        g_feat = g_mlp(g_in).max(-1)[0].unsqueeze(1)  # B,1,1024
        g_xyz = xyz2.mean(1, keepdim=True)

        f2up = self.fp3(xyz2, g_xyz, f2, g_feat)
        f1up = self.fp2(xyz1, xyz2, f1, f2up)
        f0up = self.fp1(xyz0, xyz1, None, f1up)
        return self.head(f0up.transpose(2, 1)).transpose(2, 1)  # B,N,C


def knn_graph(x, k):
    # x: B,C,N  →  idx: B,N,k
    inner = -2 * torch.bmm(x.transpose(2, 1), x)
    xx = (x**2).sum(1, keepdim=True)
    dist = xx + xx.transpose(2, 1) + inner
    return dist.topk(k, largest=False)[1]


class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):  # x: B,C,N
        B, C, N = x.shape
        idx = knn_graph(x, self.k)  # B,N,k
        base = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx_flat = (idx + base).view(B, -1)
        xt = x.transpose(2, 1).contiguous().view(B * N, C)
        neighbors = xt[idx_flat].view(B, N, self.k, C).permute(0, 3, 1, 2)  # B,C,N,k
        xi = x.unsqueeze(-1).expand_as(neighbors)
        edge = torch.cat([xi, neighbors - xi], dim=1)  # B,2C,N,k
        return self.conv(edge).max(-1)[0]  # B,out_ch,N


class DGCNN(nn.Module):
    def __init__(self, num_classes, k=20):
        super().__init__()
        self.ec1 = EdgeConv(3, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 64, k)
        self.ec4 = EdgeConv(64, 128, k)
        self.conv = nn.Sequential(
            nn.Conv1d(320, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2)
        )
        self.seg = nn.Sequential(
            nn.Conv1d(1344, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, x):  # x: B,N,3
        x = x.transpose(2, 1)  # B,3,N
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)
        cat = torch.cat([x1, x2, x3, x4], 1)  # B,320,N
        g = self.conv(cat)
        gm = g.max(-1)[0].unsqueeze(-1).expand(-1, -1, x.size(-1))  # B,1024,N
        out = self.seg(torch.cat([cat, gm], 1))  # B,C,N
        return out.transpose(2, 1)  # B,N,C


def get_model(name, num_classes):
    return {
        "PointNet": PointNet,
        "PointNet++": PointNetPP,
        "DGCNN": DGCNN,
    }[name](num_classes)
