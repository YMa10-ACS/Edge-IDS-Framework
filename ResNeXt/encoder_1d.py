"""
Description: ResNeXt1D Encoder for flow-level/tabular features
Date: 2026-03-21
Author: Yaoquan Ma
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
        self.embedding_dim = int(embedding_dim)
        self.batch_size = int(batch_size)
        self.device = device if device is not None else torch.device("cpu")

        self.model = ResNeXt1DBackbone(
            embedding_dim=self.embedding_dim,
            cut_at=cut_at,
            groups=groups,
            width_per_group=width_per_group,
        ).to(self.device).eval()

    def fit(
        self,
        X,
        y,
        epochs=3,
        batch_size=2048,
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        verbose=True,
    ):
        x_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y).reshape(-1)

        _, y_idx = np.unique(y_np, return_inverse=True)
        n_classes = len(np.unique(y_idx))

        x_tensor = torch.from_numpy(x_np).unsqueeze(1)
        y_tensor = torch.from_numpy(y_idx.astype(np.int64))
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=int(batch_size),
            shuffle=True,
        )

        cls_head = nn.Linear(self.embedding_dim, n_classes).to(self.device)
        optimizer = torch.optim.SGD(
            list(self.model.parameters()) + list(cls_head.parameters()),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(epochs), 1)
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        cls_head.train()

        last_loss = None
        last_acc = None
        for epoch in range(1, int(epochs) + 1):
            total = 0
            correct = 0
            total_loss = 0.0

            for xb, yb in loader:
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)
                emb = self.model(xb)
                logits = cls_head(emb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                total += bs
                total_loss += float(loss.item()) * bs
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())

            scheduler.step()

            last_loss = total_loss / max(total, 1)
            last_acc = correct / max(total, 1)
            if verbose:
                print(
                    f"[TRAIN][RN1D] epoch={epoch:03d}/{int(epochs):03d} "
                    f"loss={last_loss:.6f} acc={last_acc:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        self.model.eval()
        return {
            "epochs": int(epochs),
            "final_loss": float(last_loss if last_loss is not None else 0.0),
            "final_train_acc": float(last_acc if last_acc is not None else 0.0),
            "n_classes": int(n_classes),
        }

    @torch.inference_mode()
    def forward(self, X):
        x = np.asarray(X, dtype=np.float32)

        # [B, D] -> [B, 1, D]
        x_tensor = torch.from_numpy(x).unsqueeze(1).to(self.device)

        out = []
        for start in range(0, len(x_tensor), self.batch_size):
            batch = x_tensor[start : start + self.batch_size]
            emb = self.model(batch)
            out.append(emb.cpu())
        embedding = torch.cat(out, dim=0).numpy().astype(np.float32)
        return embedding
