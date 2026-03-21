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

from support import (
    load_dataset,
    transfer_embedding,
    data_preprocess,
    perf_counter,
    build_local_metrics,
    merge_transfer_metrics,
    merge_cloud_metrics,
    append_metrics_csv,
)
from monitor import read_process_cpu_rss, ProcessSampler
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
        "pca4": PCAEncoder(n_components=4),
        "pca8": PCAEncoder(n_components=8),
        "pca12": PCAEncoder(n_components=12),
        "pca16": PCAEncoder(n_components=16),
        "pca20": PCAEncoder(n_components=20),
        "pca24": PCAEncoder(n_components=24),
        "pca28": PCAEncoder(n_components=28),
        "pca32": PCAEncoder(n_components=32),
        "dnn4" : DNNEncoder(embedding_dim=4, device=device),
        "dnn8" : DNNEncoder(embedding_dim=8, device=device),
        "dnn12" : DNNEncoder(embedding_dim=12, device=device),
        "dnn16" : DNNEncoder(embedding_dim=16, device=device),
        "dnn20" : DNNEncoder(embedding_dim=20, device=device),
        "dnn24" : DNNEncoder(embedding_dim=24, device=device),
        "dnn28" : DNNEncoder(embedding_dim=28, device=device),
        "dnn32" : DNNEncoder(embedding_dim=32, device=device),
        "resnext4": RNEncoder(embedding_dim=4, device=device),
        "resnext8": RNEncoder(embedding_dim=8, device=device),
        "resnext12": RNEncoder(embedding_dim=12, device=device),
        "resnext16": RNEncoder(embedding_dim=16, device=device),
        "resnext20": RNEncoder(embedding_dim=20, device=device),
        "resnext24": RNEncoder(embedding_dim=24, device=device),
        "resnext28": RNEncoder(embedding_dim=28, device=device),
        "resnext32": RNEncoder(embedding_dim=32, device=device),
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
    embedding = model.forward(X)
    labels = y.to_numpy(dtype=np.float32).reshape(-1, 1)
    embedding_with_label = np.concatenate([embedding, labels], axis=1).astype(np.float32)
    return embedding_with_label


def encode_features_in_chunks(model, X, y, num_chunks=10):
    idx_chunks = np.array_split(np.arange(X.shape[0]), num_chunks)
    chunk_embeddings = []
    total_t0 = time.perf_counter()
    total_rows = 0

    for i, idx in enumerate(idx_chunks, start=1):
        X_chunk = X[idx]
        y_chunk = y.iloc[idx].reset_index(drop=True)

        t0 = time.perf_counter()
        emb_chunk = encode_features(model, X_chunk, y_chunk)
        dt = time.perf_counter() - t0

        total_rows += len(idx)
        chunk_embeddings.append(emb_chunk)
        print(
            f"[CHUNK {i:02d}] rows={len(idx)} "
            f"emb_shape={emb_chunk.shape} time={dt:.4f}s"
        )

    embedding = np.concatenate(chunk_embeddings, axis=0).astype(np.float32)
    metadata = {
        "shape": list(embedding.shape[:-1] + (embedding.shape[1] - 1,)),
        "dtype": "float32",
    }

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
    parser.add_argument("--run-id", default="")
    parser.add_argument("--metrics-csv", default="")
    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)
    print("Using encoder:", args.encoder)

    df = load_dataset(args.dataset, args.percentage)
    y = df["Attack_label"].copy()
    X = data_preprocess(df)
    model = encode_prepare(X, y, args.encoder, device)

    pid = os.getpid()
    pre_stats = read_process_cpu_rss(pid)
    rss_before_mb = pre_stats[1] if pre_stats is not None else 0.0

    sampler = ProcessSampler(pid=pid, interval=0.5)
    encode_t0 = time.perf_counter()
    sampler.start()
    embedding, metadata = encode_features_in_chunks(model, X, y, num_chunks=10)
    sampler.stop()
    encode_duration_s = time.perf_counter() - encode_t0

    metrics = build_local_metrics(
        run_id=args.run_id,
        encoder=args.encoder,
        embedding=embedding,
        metadata=metadata,
        encode_duration_s=encode_duration_s,
        rss_before_mb=rss_before_mb,
        sampler_samples=sampler.samples,
    )

    response, transfer_metrics = transfer_embedding(embedding, metadata)
    metrics = merge_transfer_metrics(metrics, transfer_metrics)
    metrics = merge_cloud_metrics(metrics, response)

    append_metrics_csv(metrics, csv_path=(args.metrics_csv or None))
    print("[ENCODER_METRICS] " + json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main()
