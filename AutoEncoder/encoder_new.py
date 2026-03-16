"""
Description: Trainable DNN encoder for flow-level features
Date: 2026-03-15
Author: Yaoquan Ma
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class DNNEncoder(nn.Module):
    def __init__(self, input_dim=36, embedding_dim=24, hidden_dims=(64, 32), device=None):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = tuple(hidden_dims)
        self.device = self._resolve_device(device)

        h1, h2 = self.hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, embedding_dim),
        ).to(self.device)

        self._label_encoder = None
        self._train_info = {}

    @staticmethod
    def _resolve_device(device):
        if device is None:
            return torch.device("cpu")
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            if device == "mps" and not torch.backends.mps.is_available():
                return torch.device("cpu")
            if device == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return torch.device(device)
        raise TypeError("device must be None, str, or torch.device")

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def fit(
        self,
        X,
        y,
        epochs=10,
        batch_size=2048,
        lr=1e-3,
        weight_decay=0.0,
        verbose=True,
        cache_config_path=None,
        force_retrain=False,
    ):
        if cache_config_path and (not force_retrain) and os.path.exists(cache_config_path):
            self._load_artifacts_into_self(cache_config_path)
            if verbose:
                print(f"[TRAIN] loaded pretrained encoder from: {cache_config_path}")
            return self._train_info

        X_np = self._to_numpy(X).astype(np.float32)
        y_np = self._to_numpy(y)

        if X_np.ndim != 2:
            raise ValueError(f"Expected X shape [N, D], got {X_np.shape}")
        if X_np.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got D={X_np.shape[1]}")
        if len(X_np) != len(y_np):
            raise ValueError("X and y must have the same length")

        self._label_encoder = LabelEncoder()
        y_idx = self._label_encoder.fit_transform(y_np).astype(np.int64)
        n_classes = len(self._label_encoder.classes_)
        if n_classes < 2:
            raise ValueError("Need at least 2 classes to train encoder")

        x_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_idx)
        loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

        clf_head = nn.Linear(self.embedding_dim, n_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.net.parameters()) + list(clf_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.train()
        clf_head.train()
        last_loss = None
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total = 0
            correct = 0

            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                z = self.net(xb)
                logits = clf_head(z)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                bs = yb.size(0)
                total_loss += loss.item() * bs
                total += bs
                correct += (logits.argmax(dim=1) == yb).sum().item()

            last_loss = total_loss / max(total, 1)
            acc = correct / max(total, 1)
            if verbose:
                print(f"[TRAIN] epoch={epoch:03d}/{epochs:03d} loss={last_loss:.6f} acc={acc:.4f}")

        self.eval()
        self._train_info = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "n_samples": int(len(X_np)),
            "n_classes": int(n_classes),
            "final_loss": float(last_loss if last_loss is not None else 0.0),
        }
        return self._train_info

    def _load_artifacts_into_self(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_path = config["model_path"]
        ckpt = torch.load(model_path, map_location=self.device)

        ckpt_input_dim = int(ckpt.get("input_dim", self.input_dim))
        ckpt_embedding_dim = int(ckpt.get("embedding_dim", self.embedding_dim))
        ckpt_hidden_dims = tuple(ckpt.get("hidden_dims", list(self.hidden_dims)))
        if (
            ckpt_input_dim != self.input_dim
            or ckpt_embedding_dim != self.embedding_dim
            or ckpt_hidden_dims != self.hidden_dims
        ):
            raise ValueError(
                "Checkpoint architecture mismatch. "
                f"ckpt=({ckpt_input_dim}, {ckpt_hidden_dims}, {ckpt_embedding_dim}) "
                f"current=({self.input_dim}, {self.hidden_dims}, {self.embedding_dim})"
            )

        self.load_state_dict(ckpt["state_dict"])
        self.eval()

        label_classes = ckpt.get("label_classes")
        if label_classes is not None:
            self._label_encoder = LabelEncoder()
            self._label_encoder.classes_ = np.asarray(label_classes, dtype=object)
        else:
            self._label_encoder = None
        self._train_info = ckpt.get("train_info", {})

    @torch.no_grad()
    def forward(self, x):
        x_np = self._to_numpy(x).astype(np.float32)
        if x_np.ndim != 2:
            raise ValueError(f"Expected 2D input [N, D], got shape={x_np.shape}")
        if x_np.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got D={x_np.shape[1]}")

        x_tensor = torch.from_numpy(x_np).to(self.device)
        embedding = self.net(x_tensor).cpu().numpy().astype(np.float32)
        metadata = {
            "shape": list(embedding.shape),
            "dtype": str(embedding.dtype),
        }
        return embedding, metadata

    def save_artifacts(self, model_path, config_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        ckpt = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": list(self.hidden_dims),
            "label_classes": (
                self._label_encoder.classes_.tolist() if self._label_encoder is not None else None
            ),
            "train_info": self._train_info,
        }
        torch.save(ckpt, model_path)

        config = {
            "model_path": model_path,
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": list(self.hidden_dims),
            "device": str(self.device),
            "train_info": self._train_info,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return {"model_path": model_path, "config_path": config_path}

    @classmethod
    def load_from_config(cls, config_path, device=None):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(
            input_dim=int(config["input_dim"]),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dims=tuple(config["hidden_dims"]),
            device=device if device is not None else config.get("device"),
        )

        ckpt = torch.load(config["model_path"], map_location=model.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        label_classes = ckpt.get("label_classes")
        if label_classes is not None:
            model._label_encoder = LabelEncoder()
            model._label_encoder.classes_ = np.asarray(label_classes, dtype=object)
        model._train_info = ckpt.get("train_info", {})
        return model
