"""
Description: Supervised DNN encoder for flow-level features
Date: 2026-03-21
Author: Yaoquan Ma
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DNNEncoder(nn.Module):
    """
    DNN encoder with supervised fit:
    - keep encoder structure unchanged
    - add a temporary linear classification head during fit(X, y)
    - forward() always returns embedding
    """

    def __init__(self, input_dim=36, embedding_dim=24, hidden_dims=(64, 32), device=None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.device = device if device is not None else torch.device("cpu")

        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, self.embedding_dim),
        ).to(self.device)

    def fit(
        self,
        X,
        y,
        epochs=20,
        batch_size=2048,
        lr=1e-3,
        verbose=True,
    ):
        x_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y).reshape(-1)
        _, y_idx = np.unique(y_np, return_inverse=True)
        n_classes = len(np.unique(y_idx))

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_idx.astype(np.int64))
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=int(batch_size),
            shuffle=True,
        )

        cls_head = nn.Linear(self.embedding_dim, n_classes).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.net.parameters()) + list(cls_head.parameters()),
            lr=float(lr),
        )
        criterion = nn.CrossEntropyLoss()

        self.net.train()
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
                emb = self.net(xb)
                logits = cls_head(emb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                total += bs
                total_loss += float(loss.item()) * bs
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == yb).sum().item())

            last_loss = total_loss / max(total, 1)
            last_acc = correct / max(total, 1)
            if verbose:
                print(
                    f"[TRAIN][DNN] epoch={epoch:03d}/{int(epochs):03d} "
                    f"loss={last_loss:.6f} acc={last_acc:.4f}"
                )

        self.net.eval()
        return {
            "epochs": int(epochs),
            "final_loss": float(last_loss if last_loss is not None else 0.0),
            "final_train_acc": float(last_acc if last_acc is not None else 0.0),
            "n_classes": int(n_classes),
        }

    @torch.inference_mode()
    def forward(self, x):
        x_np = np.asarray(x, dtype=np.float32)
        x_tensor = torch.from_numpy(x_np).to(self.device, dtype=torch.float32)
        embedding = self.net(x_tensor).cpu().numpy().astype(np.float32)
        return embedding
