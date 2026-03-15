#!/usr/bin/env python3
"""
Benchmark encode_features by splitting dataset into chunks.
"""

import argparse
import os
import sys
import time

import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Framework.edge import DATASET, encode_features, get_device
from Framework.support import data_preprocess, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--encoder", default="dnn32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--percentage", type=float, default=1.0)
    parser.add_argument("--chunks", type=int, default=10)
    args = parser.parse_args()

    if args.chunks <= 0:
        raise ValueError("--chunks must be > 0")

    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Encoder: {args.encoder}")
    print(f"Percentage: {args.percentage}")
    print(f"Chunks: {args.chunks}")

    df = load_dataset(args.dataset, args.percentage)
    y_all = df["Attack_label"].copy().reset_index(drop=True)
    X_all = data_preprocess(df)

    n = X_all.shape[0]
    indices = np.array_split(np.arange(n), args.chunks)

    total_rows = 0
    total_encode_sec = 0.0

    for i, idx in enumerate(indices, start=1):
        if len(idx) == 0:
            print(f"[CHUNK {i:02d}] rows=0 skip")
            continue

        X_chunk = X_all[idx]
        y_chunk = y_all.iloc[idx].reset_index(drop=True)

        t0 = time.perf_counter()
        embedding, metadata = encode_features(X_chunk, y_chunk, args.encoder, device)
        dt = time.perf_counter() - t0

        total_rows += len(idx)
        total_encode_sec += dt
        print(
            f"[CHUNK {i:02d}] rows={len(idx)} "
            f"emb_shape={embedding.shape} meta_shape={metadata.get('shape')} "
            f"time={dt:.4f}s"
        )

    print("\n=== Summary ===")
    print(f"total_rows={total_rows}")
    print(f"total_encode_time={total_encode_sec:.4f}s")
    print(
        f"avg_time_per_chunk={total_encode_sec / max(args.chunks, 1):.4f}s "
        f"avg_time_per_1k_rows={total_encode_sec / max(total_rows, 1) * 1000:.6f}s"
    )


if __name__ == "__main__":
    main()
