"""
Description: ResNeXt1D Encoder for flow-level/tabular features
Date: 2026-03-21
Author: Yaoquan Ma
"""

import numpy as np
import torch
import torch.nn as nn


class ResNeXt1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=8, width_per_group=4):
        super().__init__()
        mid_channels = groups * width_per_group

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = nn.Conv1d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(mid_channels)

        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNeXt1DBackbone(nn.Module):
    def __init__(self, embedding_dim=16, cut_at="layer2", groups=8, width_per_group=4):
        super().__init__()
        self.cut_at = cut_at

        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.Sequential(
            ResNeXt1DBlock(64, 128, stride=1, groups=groups, width_per_group=width_per_group),
            ResNeXt1DBlock(128, 128, stride=1, groups=groups, width_per_group=width_per_group),
        )
        self.layer2 = nn.Sequential(
            ResNeXt1DBlock(128, 256, stride=2, groups=groups, width_per_group=width_per_group),
            ResNeXt1DBlock(256, 256, stride=1, groups=groups, width_per_group=width_per_group),
        )

        out_channels = 128 if cut_at == "layer1" else 256
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(out_channels, embedding_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        if self.cut_at == "layer2":
            x = self.layer2(x)
        return self.head(x)


class RN1DEncoder:
    """
    ResNeXt1D-based encoder for flow-level/tabular features.
    Input X: [n_samples, n_features]
    Output embedding: [n_samples, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim=16,
        cut_at="layer2",
        groups=8,
        width_per_group=4,
        batch_size=512,
        device=None,
    ):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cpu")

        self.model = ResNeXt1DBackbone(
            embedding_dim=embedding_dim,
            cut_at=cut_at,
            groups=groups,
            width_per_group=width_per_group,
        ).to(self.device).eval()

    @torch.no_grad()
    def forward(self, X):
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input [n_samples, n_features], got shape={x.shape}")

        # [B, D] -> [B, 1, D]
        x_tensor = torch.from_numpy(x).unsqueeze(1).to(self.device)

        out = []
        for start in range(0, len(x_tensor), self.batch_size):
            batch = x_tensor[start : start + self.batch_size]
            emb = self.model(batch)
            out.append(emb.cpu())
        embedding = torch.cat(out, dim=0).numpy().astype(np.float32)
        return embedding
