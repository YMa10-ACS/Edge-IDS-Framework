"""
Description: Trainable AutoEncoder-style encoder for flow-level features
Date: 2026-03-18
Author: Yaoquan Ma
"""

from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DNNEncoder(nn.Module):
    """
    Backward-compatible class name used by edge.py.
    Internally trained as an AutoEncoder (reconstruction objective only).
    """

    def __init__(self, input_dim=36, embedding_dim=24, hidden_dims=(64, 32), device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.device = device if device is not None else torch.device("cpu")

        h1, h2 = self.hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, self.embedding_dim),
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, self.input_dim),
        ).to(self.device)

        self._train_info = {}

    def fit(
        self,
        X,
        y=None,  # kept for compatibility with edge.py calling fit(X, y)
        epochs=20,
        batch_size=2048,
        lr=1e-3,
        verbose=True,
    ):
        X_np = X.astype(np.float32, copy=False)

        x_tensor = torch.from_numpy(X_np)
        loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr,
        )

        self.train()
        last_loss = None
        for epoch in range(1, int(epochs) + 1):
            total_loss = 0.0
            total = 0
            for (xb,) in loader:
                xb = xb.to(self.device, dtype=torch.float32)
                optimizer.zero_grad()
                z = self.encoder(xb)
                x_hat = self.decoder(z)
                loss = criterion(x_hat, xb)
                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                total += bs
                total_loss += loss.item() * bs

            last_loss = total_loss / max(total, 1)
            if verbose:
                print(f"[TRAIN][AE] epoch={epoch:03d}/{epochs:03d} recon_loss={last_loss:.6f}")

        self.eval()
        self._train_info = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "objective": "reconstruction_mse",
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "n_samples": int(X_np.shape[0]),
            "final_recon_loss": float(last_loss if last_loss is not None else 0.0),
        }
        return self._train_info

    @torch.no_grad()
    def forward(self, x):
        x_np = x.astype(np.float32, copy=False)

        x_tensor = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        embedding = self.encoder(x_tensor).cpu().numpy().astype(np.float32)
        metadata = {
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
        }
        return embedding, metadata

    @torch.no_grad()
    def reconstruct(self, x):
        x_np = x.astype(np.float32, copy=False)

        x_tensor = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        x_hat = self.decoder(self.encoder(x_tensor)).cpu().numpy().astype(np.float32)
        return x_hat
