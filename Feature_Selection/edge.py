#!/usr/bin/env python
'''
Description: 
Date: 2026-02-20 00:22:01
Author: Yaoquan Ma
'''

import os

import torch
import argparse

from support import load_dataset, transfer_embedding, perf_counter

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

def feature_selection(X, y, k=6):
    selector = SelectKBest(mutual_info_classif, k=k)

    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    print("Selected features:", selected_features)

    return X_selected

@perf_counter
def pre_dataprocess(df) :
    y = df["Attack_label"]
    X = df.drop(columns=["Attack_label"])

    for col in df.columns :
        if df[col].dtype == "object" :
            print(f"col = {col}, type = {df[col].dtype}, sample = {df[col].iloc[0]}")

    X_selected = feature_selection(X, y)

    # scaling
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_selected)

    # embedding = X_scaled.astype("float32")
    embedding = X_selected.astype("float32")

    metadata = {
        "shape": list(embedding.shape),
        "dtype": str("float32")
    }

    return embedding, metadata



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset/Edge-IIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL"))
DATASET = os.path.join(DATASET_PATH, "ML-EdgeIIoT-dataset.csv") # "DNN-EdgeIIoT-dataset.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--percentage", default=1.0)
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()

    #1. Determine device 
    if args.device == "cpu" : 
        device = "cpu"
    elif args.device == "mps" :
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    elif args.device == "cuda" :
        device = "cuda" if torch.cuda.is_avialble() else "cpu"
    else :
        raise ValueError("Device must be cpu, mps, or cuda")
    
    print("Using device:", device)

    # 2. Load specific model
    # Load model to device
    # model.eval().to(device)

    # 3. Load dataset
    df = load_dataset(args.dataset, args.percentage)

    # 4. Pre-dataprocessing, autoEncoder or other
    embedding, metadata = pre_dataprocess(df)

    # 5. Transfer embedding
    transfer_embedding(embedding, metadata)
    

if __name__ == "__main__":
    main()

