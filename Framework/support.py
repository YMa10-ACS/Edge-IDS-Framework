'''
Description: 
Date: 2026-03-05 11:45:37
Author: Yaoquan Ma
'''
import os
import json
import time
import threading
from functools import wraps

import pandas as pd
import numpy as np
import requests
import csv

import ipaddress
from sklearn.preprocessing import StandardScaler




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
    if hasattr(embedding, "detach"):  # torch.Tensor
        embedding = embedding.detach().cpu().numpy()

    metadata = dict(metadata)
    metadata["shape"] = list(embedding.shape)
    metadata["dtype"] = "float32"
    payload = embedding.tobytes()
    headers = {"Meta": json.dumps(metadata)}

    try:
        resp = requests.post(
            "http://127.0.0.1:8000",
            data=payload,
            headers=headers,
            timeout=3600,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"status_code": resp.status_code, "text": resp.text}
    except Exception as exc:
        print(f"[WARN] transfer failed: {exc}")
        return {"error": str(exc)}
