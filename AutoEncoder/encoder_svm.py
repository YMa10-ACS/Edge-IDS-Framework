#!/usr/bin/env python3
"""
Description: Encoder features + SVM baseline for tabular intrusion dataset
Date: 2026-03-15
Author: Yaoquan Ma
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Framework.support import data_preprocess, load_dataset


DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/Edge-IIoTset/")
DATASET = os.path.join(DATASET_PATH, "ML-EdgeIIoT-dataset 2.csv")


def get_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device must be one of: cpu, mps, cuda")


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class EncoderClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = TabularEncoder(input_dim, hidden_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits


def train_encoder_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total += yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"loss={total_loss / max(total, 1):.4f} "
            f"acc={correct / max(total, 1):.4f}"
        )


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    encoder.eval()
    tensor = torch.from_numpy(X.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)

    out = []
    for (xb,) in loader:
        xb = xb.to(device)
        z = encoder(xb)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def build_svm() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--percentage", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)
    print("Dataset:", args.dataset)

    df = load_dataset(args.dataset, args.percentage)
    y_raw = df["Attack_label"].copy()
    X = data_preprocess(df).astype(np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype(np.int64)
    class_names = [str(c) for c in label_encoder.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    print(f"X_train={X_train.shape}, X_test={X_test.shape}, classes={len(class_names)}")

    train_dataset = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = EncoderClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_size,
        latent_dim=args.latent_size,
        num_classes=len(class_names),
    ).to(device)

    train_encoder_classifier(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    Z_train = extract_features(model.encoder, X_train, args.batch_size, device)
    Z_test = extract_features(model.encoder, X_test, args.batch_size, device)

    svm = build_svm()
    svm.fit(Z_train, y_train)
    y_pred = svm.predict(Z_test)

    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n[Encoder Features + SVM] test_accuracy={test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()
