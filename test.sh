#!/usr/bin/env bash
set -euo pipefail

dataset="/Users/sunspringmark/Library/CloudStorage/OneDrive-Personal/Study/Master_Galway/ACS/Semester 2/CT5193 Case Studies in Cybersecurity Analytics/CaseStudy/Source/dataset/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv"
records_dir="records"
mkdir -p "$records_dir"
run_ts="$(date +%Y%m%d_%H%M%S)"
summary_file="${records_dir}/encoder_metrics_${run_ts}.csv"
echo "run_id,timestamp,encoder,embedding_dim,encode_duration_s,cpu_avg_pct,cpu_max_pct,rss_before_mb,rss_peak_mb,embedding_bytes,rss_net_growth_mb,encoding_overhead_excl_embedding_mb,test_accuracy,test_f1_score" > "$summary_file"

encoders=(
"feature_selection"
"pca4"
"pca8"
"pca12"
"pca16"
"pca20"
"pca24"
"pca28"
"pca32"
"dnn4"
"dnn8"
"dnn12"
"dnn16"
"dnn20"
"dnn24"
"dnn28"
"dnn32"
"resnext4"
"resnext8"
"resnext12"
"resnext16"
"resnext20"
"resnext24"
"resnext28"
"resnext32"
)

for enc in "${encoders[@]}"; do
    ts="$(date +%Y%m%d_%H%M%S)"
    record_file="${records_dir}/${enc}_${ts}.record"
    edge_log="${records_dir}/${enc}_${ts}.edge.log"

    ./Framework/edge.py \
      --encoder "$enc" \
      --dataset "$dataset" \
      --run-id "$run_ts" > "$edge_log" 2>&1 &
    EDGE_PID=$!
    python3 monitor_edge_cpu.py --pid "$EDGE_PID" --interval 1 > "$record_file" || true

    if wait "$EDGE_PID"; then
        edge_status=0
    else
        edge_status=$?
    fi

    metrics_line="$(grep -F '[ENCODER_METRICS]' "$edge_log" | tail -n1 || true)"
    if [[ $edge_status -eq 0 && -n "$metrics_line" ]]; then
        metrics_json="${metrics_line#\[ENCODER_METRICS\] }"
        csv_row="$(python3 -c '
import json, sys
d = json.loads(sys.argv[1])
cols = [
    "run_id","timestamp","encoder","embedding_dim","encode_duration_s","cpu_avg_pct",
    "cpu_max_pct","rss_before_mb","rss_peak_mb","embedding_bytes","rss_net_growth_mb",
    "encoding_overhead_excl_embedding_mb","test_accuracy","test_f1_score"
]
print(",".join(str(d.get(k, "")) for k in cols))
' "$metrics_json")"
        echo "$csv_row" >> "$summary_file"
        echo "[SUMMARY] encoder=${enc} metrics captured"
    else
        echo "${run_ts},${ts},${enc},,,,,,,,,,," >> "$summary_file"
        echo "[SUMMARY] encoder=${enc} failed(status=${edge_status}) or metrics missing: ${edge_log}"
    fi
done

echo "Saved summary to: ${summary_file}"
