#!/usr/bin/env bash
set -euo pipefail

dataset="/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv"
records_dir="records"
mkdir -p "$records_dir"
run_ts="$(date +%Y%m%d_%H%M%S)"
summary_file="${records_dir}/encoder_metrics_${run_ts}.csv"
echo "timestamp,encoder,accuracy,f1_score,exit_code,record_file,edge_log" > "$summary_file"

encoders=(
  "feature_selection"
  "pca16"
  "pca24"
  "pca32"
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
    edge_log="${records_dir}/${enc}_${ts}.edge.log"

    ./Framework/edge.py \
      --encoder "$enc" \
      --dataset "$dataset" > "$edge_log" 2>&1 &
    EDGE_PID=$!
    python3 monitor_edge_cpu.py --pid "$EDGE_PID" --interval 1 > "$record_file"
    if wait "$EDGE_PID"; then
        exit_code=0
    else
        exit_code=$?
    fi

    acc="$(grep -oE '\[CLOUD\] accuracy=[0-9.]+$' "$edge_log" | tail -n1 | cut -d= -f2 || true)"
    f1="$(grep -oE '\[CLOUD\] f1_score=[0-9.]+$' "$edge_log" | tail -n1 | cut -d= -f2 || true)"
    acc="${acc:-NA}"
    f1="${f1:-NA}"

    echo "${ts},${enc},${acc},${f1},${exit_code},${record_file},${edge_log}" >> "$summary_file"
    echo "[SUMMARY] encoder=${enc} accuracy=${acc} f1=${f1} exit_code=${exit_code}"
done

echo "Saved summary to: ${summary_file}"
