#!/usr/bin/env bash
set -euo pipefail

OUT=${1:?output csv path is required}
INTERVAL=${2:-1}

mkdir -p "$(dirname "$OUT")"
echo "timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw" > "$OUT"

while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits >> "$OUT"
  sleep "$INTERVAL"
done
