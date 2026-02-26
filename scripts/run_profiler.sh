#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${SCENARIO:-dataloader}"

python benchmarks/micro/profiler_demo.py \
  --scenario "${SCENARIO}" \
  --batch_size 64 \
  --num_workers 2 \
  --pin_memory \
  --persistent_workers \
  --amp \
  --trace_dir "results/traces/profiler_demo_${SCENARIO}" \
  --output_json "results/metrics_demo_${SCENARIO}.json"
