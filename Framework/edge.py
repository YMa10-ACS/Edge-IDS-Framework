#!/usr/bin/env python
'''
Description: 
Date: 2026-03-10 20:12:17
Author: Yaoquan Ma
'''
"""
Description: Edge encoder entrypoint
Date: 2026-03-10
Author: Yaoquan Ma
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from support import load_dataset, transfer_embedding, data_preprocess, perf_counter
from PCA.encoder import PCAEncoder
from Feature_Selection.encoder import FSEncoder
from ResNeXt.encoder import RNEncoder
from AutoEncoder.encoder_new import DNNEncoder


DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/Edge-IIoTset/")
DATASET = os.path.join(DATASET_PATH, "DNN-EdgeIIoT-dataset.csv")


def get_device(device_name):
    if device_name == "mps" and torch.backends.mps.is_available() :
        return torch.device(device_name)
    elif device_name == "cuda" and torch.cuda.is_available() :
        return torch.device(device_name)

    return torch.device("cpu")


def encode_prepare(X, y, encode_type, device):
    encoder_mapping = {
        "feature_selection": FSEncoder(),
        "pca": PCAEncoder(n_components=16),
        "dnn16" : DNNEncoder(embedding_dim=16, device=device),
        "dnn24" : DNNEncoder(embedding_dim=24, device=device),
        "dnn32" : DNNEncoder(embedding_dim=32, device=device),
        "resnext16": RNEncoder(embedding_dim=16),
        "resnext24": RNEncoder(embedding_dim=24),
        "resnext32": RNEncoder(embedding_dim=32),
    }
    if encode_type not in encoder_mapping:
        raise ValueError(
            f"Unknown encoder '{encode_type}', choose from {list(encoder_mapping.keys())}"
        )

    model = encoder_mapping[encode_type]
    fit_fn = getattr(model, "fit", None)
    if callable(fit_fn):
        fit_fn(X, y)
    return model

@perf_counter
def encode_features(model, X, y):
    embedding, metadata = model.forward(X)

    labels = y.to_numpy(dtype=np.float32).reshape(-1, 1)
    if labels is not None:
        if len(labels) != embedding.shape[0]:
            raise ValueError(
                f"labels length ({len(labels)}) must match embedding rows ({embedding.shape[0]})"
            )
    embedding_with_label = np.concatenate([embedding, labels], axis=1).astype(np.float32)

    # Ensure metadata is JSON serializable before sending as header.
    json.dumps(metadata)

    return embedding_with_label, metadata


def encode_features_in_chunks(model, X, y, num_chunks=10):
    if num_chunks <= 0:
        raise ValueError("num_chunks must be > 0")

    idx_chunks = np.array_split(np.arange(X.shape[0]), num_chunks)
    chunk_embeddings = []
    final_metadata = None
    total_t0 = time.perf_counter()
    total_rows = 0

    for i, idx in enumerate(idx_chunks, start=1):
        if len(idx) == 0:
            print(f"[CHUNK {i:02d}] rows=0 skip")
            continue

        X_chunk = X[idx]
        y_chunk = y.iloc[idx].reset_index(drop=True)

        t0 = time.perf_counter()
        emb_chunk, meta_chunk = encode_features(model, X_chunk, y_chunk)
        dt = time.perf_counter() - t0

        total_rows += len(idx)
        chunk_embeddings.append(emb_chunk)
        final_metadata = meta_chunk
        print(
            f"[CHUNK {i:02d}] rows={len(idx)} "
            f"emb_shape={emb_chunk.shape} time={dt:.4f}s"
        )

    if not chunk_embeddings:
        raise ValueError("No chunk data was encoded")

    embedding = np.concatenate(chunk_embeddings, axis=0).astype(np.float32)
    metadata = dict(final_metadata) if final_metadata is not None else {}
    metadata["shape"] = list(embedding.shape[:-1] + (embedding.shape[1] - 1,))
    metadata["dtype"] = "float32"

    total_dt = time.perf_counter() - total_t0
    avg_dt = total_dt / max(len(chunk_embeddings), 1)
    print(
        f"[CHUNK SUMMARY] chunks={len(chunk_embeddings)} rows={total_rows} "
        f"total_encode_time={total_dt:.4f}s avg_per_chunk={avg_dt:.4f}s"
    )
    return embedding, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--percentage", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder", default="pca")
    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)
    print("Using encoder:", args.encoder)

    df = load_dataset(args.dataset, args.percentage)
    y = df["Attack_label"].copy()
    X = data_preprocess(df)
    model = encode_prepare(X, y, args.encoder, device)

    embedding, metadata = encode_features_in_chunks(model, X, y, num_chunks=10)
    response = transfer_embedding(embedding, metadata)
    if isinstance(response, dict):
        if "test_accuracy" in response:
            print(f"[CLOUD] accuracy={response['test_accuracy']:.6f}")
        if "test_f1_score" in response:
            print(f"[CLOUD] f1_score={response['test_f1_score']:.6f}")
        if "error" in response:
            print(f"[CLOUD] error={response['error']}")


if __name__ == "__main__":
    main()
