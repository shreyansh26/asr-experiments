#!/usr/bin/env bash
set -euo pipefail

mkdir -p inference/results/nsys
cd /mnt/ssd1/shreyansh/home_dir/asr_experiments

SESSION=fp8static_c1_pytrace_graph
REPORT=inference/results/nsys/fp8static_c1_5s_pytrace_graph
AUDIO=data/prepared_data/carta_september_2024/call-156003_0.8311693678982316_4.230553711834489/channel_0.wav

until curl -fsS http://127.0.0.1:8090/v1/models >/dev/null; do
  sleep 2
done

echo "Running unprofiled warmups..."
for _ in 1 2 3; do
  uv run python inference/run_infer.py "$AUDIO" \
    --uniform-audio-length 5 \
    --stream \
    --no-print-text \
    --timeout-seconds 60
done

capture_started=false

cleanup() {
  if [[ "$capture_started" == true ]]; then
    echo "Stopping Nsight collection..."
    uv run nsys stop --session="$SESSION" || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Nsight collection..."
uv run nsys start --session="$SESSION" 

capture_started=true
uv run nsys sessions list

echo "Running profiled request..."
uv run python inference/run_infer.py "$AUDIO" \
  --uniform-audio-length 5 \
  --stream \
  --no-print-text \
  --timeout-seconds 60

echo "Stopping Nsight collection..."
uv run nsys stop --session="$SESSION"
capture_started=false

trap - EXIT INT TERM

echo "Report: ${REPORT}.nsys-rep"