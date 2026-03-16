#!/usr/bin/env bash
set -euo pipefail

dataset="/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv"
records_dir="records"
mkdir -p "$records_dir"

encoders=(
  "feature_selection"
  "pca"
  "dnn16"
  "dnn24"
  "dnn32"
  "resnext16"
  "resnext24"
  "resnext32"
)

for enc in "${encoders[@]}"; do
    ts="$(date +%Y%m%d_%H%M%S)"
    record_file="${records_dir}/${enc}_${ts}.record"

    ./Framework/edge.py \
      --encoder "$enc" \
      --dataset "$dataset" &
    EDGE_PID=$!
    python3 monitor_edge_cpu.py --pid "$EDGE_PID" --interval 1 > "$record_file"
    wait "$EDGE_PID"
done
