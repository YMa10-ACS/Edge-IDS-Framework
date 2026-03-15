"""
Description: ResNeXt Encoder for tabular features
Date: 2026-03-12
Author: Yaoquan Ma
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class ResNeXtBackbone(nn.Module):
    def __init__(self, embedding_dim=16, cut_at="layer1", use_pretrained=False):
        super().__init__()
        
        base = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False,
        )
        with torch.no_grad():
            if old_conv1.weight.shape[1] == 3:
                base.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.cut_at = cut_at

        in_channels = 256 if cut_at == "layer1" else 512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, embedding_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        if self.cut_at == "layer2":
            x = self.layer2(x)
        return self.head(x)


class RNEncoder:
    """
    ResNeXt-based encoder for tabular data.
    Supports output embedding dimension 16 or 32.
    """

    def __init__(
        self,
        embedding_dim=16,
        image_size=8,
        cut_at="layer1",
        use_pretrained=False,
        batch_size=256,
        device=None,
    ):
        if embedding_dim not in [ 16, 24, 32 ]:
            raise ValueError("embedding_dim must be 16 24, or 32")
        if image_size < 8:
            raise ValueError("image_size must be >= 8")

        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.target_len = image_size * image_size
        self.batch_size = batch_size
        self.device = device

        self.model = ResNeXtBackbone(
            embedding_dim=embedding_dim,
            cut_at=cut_at,
            use_pretrained=use_pretrained,
        ).to(self.device).eval()

    def pseudo_image(self, X):
        n, d = X.shape
        if d < self.target_len:
            X = np.pad(X, ((0, 0), (0, self.target_len - d)), mode="constant")
        elif d > self.target_len:
            X = X[:, : self.target_len]
        return X.reshape(n, 1, self.image_size, self.image_size).astype(np.float32)

    @torch.no_grad()
    def forward(self, X):

        imgs = self.pseudo_image(X)
        x_tensor = torch.from_numpy(imgs).to(self.device)

        out = []
        for start in range(0, len(x_tensor), self.batch_size):
            batch = x_tensor[start : start + self.batch_size]
            emb = self.model(batch)
            out.append(emb.cpu())
        embedding = torch.cat(out, dim=0).numpy().astype(np.float32)

        metadata = {
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
        }
        return embedding, metadata
