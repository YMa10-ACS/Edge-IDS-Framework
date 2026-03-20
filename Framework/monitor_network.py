"""
Network transfer monitoring utilities for embedding upload.
"""

import json
import time

import requests


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

    transfer_metrics contains both:
      - estimated app-layer request bytes
      - network-level bytes for the given TCP port (if scapy is available)
    """
    if hasattr(embedding, "detach"):  # torch.Tensor
        embedding = embedding.detach().cpu().numpy()

    metadata = dict(metadata)
    metadata["shape"] = list(embedding.shape)
    metadata["dtype"] = "float32"

    payload = embedding.tobytes()
    meta_json = json.dumps(metadata)
    headers = {"Meta": meta_json}

    meta_bytes = meta_json.encode("utf-8")
    payload_bytes = len(payload)
    metadata_bytes = len(meta_bytes)
    estimated_request_bytes = payload_bytes + metadata_bytes

    network_stats = {
        "network_capture_enabled": False,
        "network_tx_bytes": 0,
        "network_rx_bytes": 0,
        "network_packets": 0,
    }
    sniffer = None

    try:
        from scapy.all import AsyncSniffer, IP, TCP  # type: ignore

        def on_packet(pkt):
            if IP not in pkt or TCP not in pkt:
                return
            tcp_layer = pkt[TCP]
            if tcp_layer.sport != port and tcp_layer.dport != port:
                return
            ip_len = int(pkt[IP].len)
            network_stats["network_packets"] += 1
            # client -> server
            if tcp_layer.dport == port:
                network_stats["network_tx_bytes"] += ip_len
            # server -> client
            else:
                network_stats["network_rx_bytes"] += ip_len

        sniffer = AsyncSniffer(
            iface=iface,
            filter=f"tcp port {port}",
            prn=on_packet,
            store=False,
        )
        sniffer.start()
        network_stats["network_capture_enabled"] = True
    except Exception:
        # Optional capture: keep function usable without scapy/root permissions.
        network_stats["network_capture_enabled"] = False

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, data=payload, headers=headers, timeout=timeout)
        transfer_duration_s = time.perf_counter() - t0
        resp.raise_for_status()
        try:
            response_data = resp.json()
        except ValueError:
            response_data = {"status_code": resp.status_code, "text": resp.text}
    except Exception as exc:
        transfer_duration_s = time.perf_counter() - t0
        response_data = {"error": str(exc)}
        print(f"[WARN] transfer failed: {exc}")
    finally:
        if sniffer is not None:
            try:
                sniffer.stop()
            except Exception:
                pass

    network_total_bytes = int(
        network_stats["network_tx_bytes"] + network_stats["network_rx_bytes"]
    )
    transfer_metrics = {
        "payload_bytes": payload_bytes,
        "metadata_bytes": metadata_bytes,
        "estimated_request_bytes": estimated_request_bytes,
        "estimated_request_mb": round(estimated_request_bytes / (1024.0 * 1024.0), 6),
        "transfer_duration_s": round(float(transfer_duration_s), 6),
        **network_stats,
        "network_total_bytes": network_total_bytes,
        "network_total_mb": round(network_total_bytes / (1024.0 * 1024.0), 6),
    }
    return response_data, transfer_metrics
