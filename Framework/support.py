'''
Description: 
Date: 2026-03-05 11:45:37
Author: Yaoquan Ma
'''
import os
import json
import time
from functools import wraps
from datetime import datetime

import pandas as pd
import numpy as np
import csv

import ipaddress
from sklearn.preprocessing import StandardScaler
from monitor_network import monitored_transfer_embedding

TRANSFER_METRIC_FIELDS = [
    "payload_bytes",
    "metadata_bytes",
    "estimated_request_mb",
    "transfer_duration_s",
    "network_tx_bytes",
    "network_total_bytes",
    "network_total_mb",
]

METRICS_FIELDS = [
    "run_id",
    "timestamp",
    "encoder",
    "embedding_dim",
    "encode_duration_s",
    "cpu_avg_pct",
    "cpu_max_pct",
    "rss_before_mb",
    "rss_peak_mb",
    "embedding_bytes",
    "rss_net_growth_mb",
    *TRANSFER_METRIC_FIELDS,
    "test_accuracy",
    "test_f1_score",
]


def build_local_metrics(
    run_id,
    encoder,
    embedding,
    metadata,
    encode_duration_s,
    rss_before_mb,
    sampler_samples,
):
    cpu_samples = [cpu for cpu, _ in sampler_samples]
    rss_samples = [rss for _, rss in sampler_samples]
    cpu_avg_pct = float(np.mean(cpu_samples)) if cpu_samples else 0.0
    cpu_max_pct = float(np.max(cpu_samples)) if cpu_samples else 0.0
    rss_peak_mb = float(np.max(rss_samples)) if rss_samples else float(rss_before_mb)

    emb_shape = metadata.get("shape", [])
    if isinstance(emb_shape, (list, tuple)) and len(emb_shape) >= 2:
        embedding_dim = int(emb_shape[1])
        embedding_rows = int(emb_shape[0])
    else:
        embedding_dim = int(embedding.shape[1] - 1)
        embedding_rows = int(embedding.shape[0])

    embedding_bytes = int(embedding_rows * embedding_dim * np.dtype(np.float32).itemsize)
    rss_net_growth_mb = float(rss_peak_mb - float(rss_before_mb))

    return {
        "run_id": str(run_id),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "encoder": str(encoder),
        "embedding_dim": embedding_dim,
        "encode_duration_s": round(float(encode_duration_s), 6),
        "cpu_avg_pct": round(cpu_avg_pct, 4),
        "cpu_max_pct": round(cpu_max_pct, 4),
        "rss_before_mb": round(float(rss_before_mb), 4),
        "rss_peak_mb": round(float(rss_peak_mb), 4),
        "embedding_bytes": embedding_bytes,
        "rss_net_growth_mb": round(rss_net_growth_mb, 4),
        "test_accuracy": "",
        "test_f1_score": "",
    }


def merge_cloud_metrics(metrics, response):
    merged = dict(metrics)
    if not isinstance(response, dict):
        return merged

    if "test_accuracy" in response:
        print(f"[CLOUD] accuracy={response['test_accuracy']:.4f}")
        merged["test_accuracy"] = round(float(response["test_accuracy"]), 6)
    if "test_f1_score" in response:
        print(f"[CLOUD] f1_score={response['test_f1_score']:.4f}")
        merged["test_f1_score"] = round(float(response["test_f1_score"]), 6)
    if "error" in response:
        print(f"[CLOUD] error={response['error']}")
    return merged


def merge_transfer_metrics(metrics, transfer_metrics):
    merged = dict(metrics)
    if not isinstance(transfer_metrics, dict):
        return merged
    for key in TRANSFER_METRIC_FIELDS:
        if key in transfer_metrics:
            merged[key] = transfer_metrics[key]
    return merged


def append_metrics_csv(metrics, csv_path=None, records_dir="records"):
    run_id = str(metrics.get("run_id", "")).strip()
    if csv_path:
        target_path = csv_path
    else:
        suffix = run_id if run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = os.path.join(records_dir, f"encoder_metrics_{suffix}.csv")

    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    write_header = not os.path.exists(target_path) or os.path.getsize(target_path) == 0
    with open(target_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in METRICS_FIELDS})
    return target_path




def perf_counter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__}: {end - start:.2f}s")
        return ret
    return wrapper


def ipv4_to_int(x):
    s = str(x).strip()
    if s == "" or s == "0" or s == "0.0":
        return np.nan
    try:
        return int(ipaddress.IPv4Address(s))
    except ipaddress.AddressValueError:
        return np.nan

@perf_counter
def load_dataset(dataset_path, percentage) :
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with open(dataset_path) as f:
        header = f.readline()
        print(f" len = {len(header.split(","))}")
    df = pd.read_csv(dataset_path, sep=",", quoting=csv.QUOTE_MINIMAL, low_memory=False)

    if percentage < 1.0 :
        df = df.sample(frac=percentage, random_state=42)

    return df

@perf_counter
def data_preprocess(df) :

    # frame.time fromat = "2021 11:44:10.081753000" -> year hour:minute:second.nanosecond
    ts = pd.to_datetime(
            df["frame.time"],
            format="%Y %H:%M:%S.%f",
            errors="coerce",
    )
    df["frame.time"] = ts.astype("int64") / 1e9
    
    # ip.src_host = 192.168.0.128 -> 3232235648
    df["ip.src_host"] = df["ip.src_host"].apply(ipv4_to_int)

    # ip.dst_host = 192.168.0.128 -> 3232235648
    df["ip.dst_host"] = df["ip.dst_host"].apply(ipv4_to_int)

    df["tcp.srcport"] = pd.to_numeric(df["tcp.srcport"], errors="coerce") 
    df["tcp.srcport"] = df["tcp.srcport"].fillna(0).astype(int)

    # arp.src.proto_ipv4 and arp.dst.proto_ipv4 are always 0.0
    df.drop("arp.dst.proto_ipv4", axis = 1, inplace = True)
    df.drop("arp.src.proto_ipv4", axis = 1, inplace = True)

    # The element of icmp.checksum are always 0.0
    df.drop("icmp.checksum", axis = 1, inplace = True)
    df.drop("http.file_data", axis = 1, inplace = True)

    df.drop("http.request.uri.query", axis = 1, inplace = True)
    df.drop("http.request.method", axis = 1, inplace = True)
    df.drop("http.referer", axis = 1, inplace = True)
    df.drop("http.request.full_uri", axis = 1, inplace = True)
    df.drop("http.request.version", axis = 1, inplace = True)
    df.drop("tcp.payload", axis = 1, inplace = True)

    df["tcp.options"] = pd.to_numeric(df["tcp.options"], errors="coerce")
    # df.drop("dns.qry.name.len", axis = 1, inplace = True)
    
    df.drop("dns.qry.name.len", axis = 1, inplace = True)
    df.drop("dns.qry.name", axis = 1, inplace = True)

    # delete ALL mqtt columnn
    mqtt_cols = [col for col in df.columns if "mqtt" in col.lower()]
    df = df.drop(columns=mqtt_cols)

    df.drop("Attack_type", axis = 1, inplace =  True)
    df.drop("Attack_label", axis = 1, inplace =  True)

    df = df.apply(pd.to_numeric, errors="coerce")

    total_cells = df.shape[0] * df.shape[1]
    total_nan = df.isna().sum().sum()
    print(f"shape={df.shape}")
    print(f"NaN total={total_nan} ({total_nan/total_cells:.2%})")

    df = df.fillna(0.0)
    print(f"shape={df.shape}")

    print("Remaining columns:", len(df.columns))

    scaler = StandardScaler()
    df = scaler.fit_transform(df.to_numpy())

    return df

@perf_counter
def transfer_embedding(embedding, metadata):
    response_data, transfer_metrics = monitored_transfer_embedding(
        embedding=embedding,
        metadata=metadata,
        url="http://127.0.0.1:8000",
        timeout=3600,
        port=8000,
    )
    print("[TRANSFER_METRICS] " + json.dumps(transfer_metrics, ensure_ascii=True))
    return response_data, transfer_metrics
