'''
Description: 
Date: 2026-03-15 18:09:28
Author: Yaoquan Ma
'''
import torch
import torch.nn as nn
import numpy as np

class DNNEncoder(nn.Module):
    def __init__(self, input_dim=36, embedding_dim=24, hidden_dims=(64, 32), device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.device = device

        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, embedding_dim),
        ).to(self.device).eval()

    @torch.no_grad()
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x.astype(np.float32))
        elif torch.is_tensor(x):
            x_tensor = x.float()
        else:
            raise TypeError("x must be numpy.ndarray or torch.Tensor")

        if x_tensor.ndim != 2:
            raise ValueError(f"Expected 2D input [N, D], got shape={tuple(x_tensor.shape)}")
        if x_tensor.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got D={x_tensor.shape[1]}"
            )

        embedding = self.net(x_tensor.to(self.device)).cpu().numpy().astype(np.float32)
        return embedding
