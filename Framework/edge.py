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
import threading
import subprocess
from datetime import datetime

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


def read_process_cpu_rss(pid):
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "%cpu=", "-o", "rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    try:
        cpu_pct = float(parts[0])
        rss_mb = int(parts[1]) / 1024.0
    except ValueError:
        return None
    return cpu_pct, rss_mb


class ProcessSampler:
    def __init__(self, pid, interval=0.5):
        self.pid = pid
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            stats = read_process_cpu_rss(self.pid)
            if stats is not None:
                self.samples.append(stats)
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def get_device(device_name):
    if device_name == "mps" and torch.backends.mps.is_available() :
        return torch.device(device_name)
    elif device_name == "cuda" and torch.cuda.is_available() :
        return torch.device(device_name)

    return torch.device("cpu")


def encode_prepare(X, y, encode_type, device):
    encoder_mapping = {
        "feature_selection": FSEncoder(),
        "pca16": PCAEncoder(n_components=16),
        "pca24": PCAEncoder(n_components=24),
        "pca32": PCAEncoder(n_components=32),
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
    embedding_with_label = np.concatenate([embedding, labels], axis=1).astype(np.float32)
    return embedding_with_label, metadata


def encode_features_in_chunks(model, X, y, num_chunks=10):
    idx_chunks = np.array_split(np.arange(X.shape[0]), num_chunks)
    chunk_embeddings = []
    final_metadata = None
    total_t0 = time.perf_counter()
    total_rows = 0

    for i, idx in enumerate(idx_chunks, start=1):
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
    parser.add_argument("--run-id", default="")
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

    cpu_samples = [cpu for cpu, _ in sampler.samples]
    rss_samples = [rss for _, rss in sampler.samples]
    cpu_avg_pct = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    cpu_max_pct = float(np.max(cpu_samples)) if cpu_samples else 0.0
    rss_peak_mb = float(np.max(rss_samples)) if rss_samples else rss_before_mb

    emb_shape = metadata.get("shape", [])
    if isinstance(emb_shape, (list, tuple)) and len(emb_shape) >= 2:
        embedding_dim = int(emb_shape[1])
        embedding_rows = int(emb_shape[0])
    else:
        embedding_dim = int(embedding.shape[1] - 1)
        embedding_rows = int(embedding.shape[0])

    embedding_bytes = int(embedding_rows * embedding_dim * np.dtype(np.float32).itemsize)
    rss_net_growth_mb = float(rss_peak_mb - rss_before_mb)
    encoding_overhead_excl_embedding_mb = float(
        rss_net_growth_mb - (embedding_bytes / 1024.0 / 1024.0)
    )

    metrics = {
        "run_id": str(args.run_id),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "encoder": str(args.encoder),
        "embedding_dim": embedding_dim,
        "encode_duration_s": round(float(encode_duration_s), 6),
        "cpu_avg_pct": round(cpu_avg_pct, 4),
        "cpu_max_pct": round(cpu_max_pct, 4),
        "rss_before_mb": round(float(rss_before_mb), 4),
        "rss_peak_mb": round(float(rss_peak_mb), 4),
        "embedding_bytes": embedding_bytes,
        "rss_net_growth_mb": round(rss_net_growth_mb, 4),
        "encoding_overhead_excl_embedding_mb": round(
            encoding_overhead_excl_embedding_mb, 4
        ),
        "test_accuracy": "",
        "test_f1_score": "",
    }

    response = transfer_embedding(embedding, metadata)
    if isinstance(response, dict):
        if "test_accuracy" in response:
            print(f"[CLOUD] accuracy={response['test_accuracy']:.4f}")
            metrics["test_accuracy"] = round(float(response["test_accuracy"]), 6)
        if "test_f1_score" in response:
            print(f"[CLOUD] f1_score={response['test_f1_score']:.4f}")
            metrics["test_f1_score"] = round(float(response["test_f1_score"]), 6)
        if "error" in response:
            print(f"[CLOUD] error={response['error']}")
    print("[ENCODER_METRICS] " + json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main()
