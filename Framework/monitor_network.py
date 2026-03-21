'''
Description: 
Date: 2026-03-21 14:41:59
Author: Yaoquan Ma
'''
"""
Network transfer monitoring utilities for embedding upload.
Uses interface counters (netstat snapshots) instead of packet capture.
"""

import json
import subprocess
import time

import requests


def _read_interface_counters(iface):
    cmd = ["netstat", "-b", "-I", iface]
    out = subprocess.check_output(cmd, text=True)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected netstat output for iface '{iface}'.")

    for ln in lines[1:]:
        parts = ln.split()
        if not parts or parts[0] != iface or len(parts) < 11:
            continue
        try:
            return {
                "ipkts": int(parts[4]),
                "ibytes": int(parts[6]),
                "opkts": int(parts[7]),
                "obytes": int(parts[9]),
            }
        except ValueError:
            continue

    raise RuntimeError(f"Interface '{iface}' counters not found in netstat output.")


def monitored_transfer_embedding(
    embedding,
    metadata,
    url="http://127.0.0.1:8000",
    timeout=3600,
    port=8000,
    iface="lo0",
):
    """
    Returns:
      (response_data, transfer_metrics)
    Notes:
      - Uses interface-level counters (not port-level packet capture).
      - `port` is kept for call-site compatibility but not used.
    """
    del port

    if hasattr(embedding, "detach"):  # torch.Tensor
        embedding = embedding.detach().cpu().numpy()

    metadata = dict(metadata)
    metadata["shape"] = list(embedding.shape)
    metadata["dtype"] = "float32"

    payload = embedding.tobytes()
    meta_json = json.dumps(metadata)
    edge_send_start_ns = time.time_ns()
    headers = {
        "Meta": meta_json,
        "X-Edge-Send-Start-Ns": str(edge_send_start_ns),
    }

    metadata_bytes = len(meta_json.encode("utf-8"))
    payload_bytes = len(payload)
    estimated_request_bytes = payload_bytes + metadata_bytes

    before = _read_interface_counters(iface)
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        try:
            response_data = resp.json()
        except ValueError:
            response_data = {"status_code": resp.status_code, "text": resp.text}
    except Exception as exc:
        response_data = {"error": str(exc)}
        print(f"[WARN] transfer failed: {exc}")
    finally:
        transfer_duration_s = time.perf_counter() - t0
        after = _read_interface_counters(iface)
    
    print(f"before = {before}, after = {after}")

    # Interface counters: inbound ~= server->client, outbound ~= client->server.
    network_tx_bytes = max(0, after["obytes"] - before["obytes"])
    network_rx_bytes = max(0, after["ibytes"] - before["ibytes"])
    network_packets = max(0, after["opkts"] - before["opkts"]) + max(
        0, after["ipkts"] - before["ipkts"]
    )
    network_total_bytes = int(network_tx_bytes + network_rx_bytes)

    transfer_metrics = {
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "estimated_request_bytes": estimated_request_bytes,
        "estimated_request_mb": round(estimated_request_bytes / (1024.0 * 1024.0), 6),
        "transfer_duration_s": round(float(transfer_duration_s), 6),
        "network_capture_enabled": True,
        "network_tx_bytes": network_tx_bytes,
        "network_rx_bytes": network_rx_bytes,
        "network_packets": network_packets,
        "network_total_bytes": network_total_bytes,
        "network_total_mb": round(network_total_bytes / (1024.0 * 1024.0), 6),
    }
    if isinstance(response_data, dict):
        if response_data.get("cloud_receive_duration_s") is not None:
            transfer_metrics["cloud_receive_duration_s"] = float(
                response_data["cloud_receive_duration_s"]
            )
        if response_data.get("edge_to_cloud_receive_s") is not None:
            transfer_metrics["network_latency_s"] = float(
                response_data["edge_to_cloud_receive_s"]
            )
        if response_data.get("inference_duration_s") is not None:
            transfer_metrics["inference_duration_s"] = float(
                response_data["inference_duration_s"]
            )
        if response_data.get("inference_per_record_s") is not None:
            transfer_metrics["inference_per_record_s"] = float(
                response_data["inference_per_record_s"]
            )
    return response_data, transfer_metrics
