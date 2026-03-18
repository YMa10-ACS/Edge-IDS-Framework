#!/usr/bin/env python3
"""
One-shot training cloud service:
- First request: must include embedding + labels -> train once + test once.
- Later requests: embedding only -> predict only.
"""

import json
import time

import numpy as np
from flask import Flask, jsonify, request
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from support import perf_counter

app = Flask(__name__)

# Global model state in memory (single-process service).
svm_model = None
rf_model = None
validation_info = {}


def to_numpy_dtype(dtype_str):
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    return mapping.get(dtype_str, np.float32)


def load_meta():
    meta_header = request.headers.get("Meta")
    if not meta_header:
        raise ValueError("Missing metadata header: Meta")
    try:
        meta = json.loads(meta_header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Meta JSON: {exc}") from exc
    return meta


def decode_embedding(raw, meta):
    # Just for debug
    if "shape" not in meta or "dtype" not in meta:
        raise ValueError("Meta must include 'shape' and 'dtype'")

    shape = meta["shape"]
    dtype = to_numpy_dtype(str(meta["dtype"]))
    print(f"shape = {shape}, dtype = {dtype}")

    arr = np.frombuffer(raw, dtype=dtype)
    try:
        arr = arr.reshape(shape)
    except ValueError as exc:
        raise ValueError(f"Cannot reshape payload to {shape}: {exc}") from exc

    # SVM expects 2D matrix: [n_samples, n_features].
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32)


def extract_labels(X):
    y = X[:, -1].astype(np.float32)
    X = X[:, :-1]
    
    return X, y

@perf_counter
def svm_build():
    return Pipeline(
        steps=[
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    tol=1e-3,
                    random_state=42,
                    dual=False,
                ),
            ),
        ]
    )

@perf_counter
def svm_train(svm_model, X_train, y_train):
    svm_model.fit(X_train, y_train)    

@perf_counter
def svm_predict(svm_model, X) :
    return svm_model.predict(X)


@perf_counter
def rf_build():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )


@perf_counter
def rf_train(rf_model, X_train, y_train):
    rf_model.fit(X_train, y_train)


@perf_counter
def rf_predict(rf_model, X):
    return rf_model.predict(X)

@app.route("/", methods=["POST"])
def train_once_then_predict():
    """
    First request:
      - Requires Meta.labels for one-time training.
      - Runs one train/validation split and returns val_accuracy.
    Later requests:
      - labels not required.
      - Only returns predictions.
    """
    global svm_model, rf_model, validation_info
    t0 = time.perf_counter()

    try:
        meta = load_meta()
        X = decode_embedding(request.data, meta)
        X, y = extract_labels(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        print("Begin build svm model")
        svm_model = svm_build()
        print("Built svm model finish")
        svm_train(svm_model, X_train, y_train)
        print("svm fit finished!")

        y_test_pred = svm_predict(svm_model, X_test)
        print("svm model predict finished!")
        svc_acc = float(accuracy_score(y_test, y_test_pred))
        svc_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
        print(f"linear_svc acc = {svc_acc}")
        print(f"linear_svc f1 = {svc_f1}")

        print("Begin build random_forest model")
        rf_model = rf_build()
        print("Built random_forest model finish")
        rf_train(rf_model, X_train, y_train)
        print("random_forest fit finished!")

        rf_test_pred = rf_predict(rf_model, X_test)
        print("random_forest model predict finished!")
        rf_acc = float(accuracy_score(y_test, rf_test_pred))
        rf_f1 = float(f1_score(y_test, rf_test_pred, average="weighted"))
        print(f"random_forest acc = {rf_acc}")
        print(f"random_forest f1 = {rf_f1}")

        validation_info = {
            # Backward-compatible fields (use LinearSVC as default baseline)
            "test_accuracy": svc_acc,
            "test_f1_score": svc_f1,
            # Explicit per-model metrics
            "linear_svc_accuracy": svc_acc,
            "linear_svc_f1_score": svc_f1,
            "random_forest_accuracy": rf_acc,
            "random_forest_f1_score": rf_f1,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": int(X.shape[1]),
            "classes": np.unique(y).tolist(),
        }

        return jsonify(
            {
                "status": "trained_once",
                "task": "train_test",
                **validation_info,
                "elapsed_sec": round(time.perf_counter() - t0, 6),
            }
        ), 200

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
