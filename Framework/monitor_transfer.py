#!/usr/bin/env python3
"""
Utilities for monitored embedding transfer.
"""

import json
import time
import requests


def monitored_transfer_embedding(embedding, metadata, url="http://127.0.0.1:8000", timeout=3600):
    if hasattr(embedding, "detach"):  # torch.Tensor
        embedding = embedding.detach().cpu().numpy()

    metadata = dict(metadata)
    metadata["shape"] = list(embedding.shape)
    metadata["dtype"] = "float32"

    payload = embedding.tobytes()
    meta_json = json.dumps(metadata)
    meta_bytes = meta_json.encode("utf-8")

    payload_bytes = len(payload)
    metadata_bytes = len(meta_bytes)
    total_transfer_bytes = payload_bytes + metadata_bytes
    total_transfer_mb = total_transfer_bytes / (1024.0 * 1024.0)

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            data=payload,
            headers={"Meta": meta_json},
            timeout=timeout,
        )
        transfer_duration_s = time.perf_counter() - t0
        resp.raise_for_status()

        try:
            response_data = resp.json()
        except ValueError:
            response_data = {"status_code": resp.status_code, "text": resp.text}

        transfer_metrics = {
            "payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "total_transfer_bytes": total_transfer_bytes,
            "total_transfer_mb": round(total_transfer_mb, 6),
            "transfer_duration_s": round(float(transfer_duration_s), 6),
        }
        return response_data, transfer_metrics

    except Exception as exc:
        transfer_duration_s = time.perf_counter() - t0
        transfer_metrics = {
            "payload_bytes": payload_bytes,
            "metadata_bytes": metadata_bytes,
            "total_transfer_bytes": total_transfer_bytes,
            "total_transfer_mb": round(total_transfer_mb, 6),
            "transfer_duration_s": round(float(transfer_duration_s), 6),
        }
        print(f"[WARN] transfer failed: {exc}")
        return {"error": str(exc)}, transfer_metrics