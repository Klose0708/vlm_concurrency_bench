#!/usr/bin/env bash
#SBATCH -J vlm_bench
#SBATCH -o /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j.out
#SBATCH -e /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00

set -euo pipefail

module purge
module add miniconda3
source ~/.bashrc || true
conda activate "${CONDA_ENV:-vlm_bench_cu124}"

cd /data/private/"$USER"/workspace/projects/vlm_concurrency_bench

export HF_HOME=/data/private/"$USER"/workspace/cache/hf
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"/datasets
export TORCH_HOME=/data/private/"$USER"/workspace/cache/torch

MODEL_KIND=${MODEL_KIND:-bf16}
DATA=${DATA:-data/mmstar/mmstar_requests.jsonl}
DATA_LIMIT=${DATA_LIMIT:-100}
NUM_PROMPTS=${NUM_PROMPTS:-20}
MAX_TOKENS=${MAX_TOKENS:-32}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 2"}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}

case "$MODEL_KIND" in
  bf16)
    MODEL_PATH=/data/private/"$USER"/workspace/models/Qwen2.5-VL-3B-Instruct
    SERVED_NAME=qwen2_5_vl_3b_bf16
    EXTRA_ARGS=(--dtype bfloat16)
    ;;
  awq)
    MODEL_PATH=/data/private/"$USER"/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ
    SERVED_NAME=qwen2_5_vl_3b_awq
    EXTRA_ARGS=(--dtype half)
    ;;
  bf16_fp8kv)
    MODEL_PATH=/data/private/"$USER"/workspace/models/Qwen2.5-VL-3B-Instruct
    SERVED_NAME=qwen2_5_vl_3b_bf16_fp8kv
    EXTRA_ARGS=(--dtype bfloat16 --kv-cache-dtype fp8)
    ;;
  awq_fp8kv)
    MODEL_PATH=/data/private/"$USER"/workspace/models/Qwen2.5-VL-3B-Instruct-AWQ
    SERVED_NAME=qwen2_5_vl_3b_awq_fp8kv
    EXTRA_ARGS=(--dtype half --kv-cache-dtype fp8)
    ;;
  *)
    echo "Unknown MODEL_KIND=$MODEL_KIND"
    exit 1
    ;;
esac

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path does not exist: $MODEL_PATH"
  exit 2
fi

mkdir -p data/mmstar results logs
if [[ ! -f "$DATA" ]]; then
  python scripts/prepare_mmstar.py --out_dir "$(dirname "$DATA")" --limit "$DATA_LIMIT"
fi

chmod +x scripts/gpu_monitor.sh

PREFIX_CACHE_ARGS=()
if vllm serve --help 2>&1 | grep -q -- "--no-enable-prefix-caching"; then
  PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi

EAGER_ARGS=()
if [[ "$ENFORCE_EAGER" = "1" ]]; then
  EAGER_ARGS=(--enforce-eager)
fi

SERVER_LOG=logs/"${SLURM_JOB_ID}_${MODEL_KIND}_server.log"
GPU_LOG=logs/"${SLURM_JOB_ID}_${MODEL_KIND}_gpu.csv"

bash scripts/gpu_monitor.sh "$GPU_LOG" 1 &
MONITOR_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting vLLM server on $(hostname), MODEL_KIND=$MODEL_KIND"
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt image=1 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --disable-log-requests \
  "${PREFIX_CACHE_ARGS[@]}" \
  "${EAGER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for server health..."
python - <<PY
import json
import time
import urllib.request

url = "http://127.0.0.1:${PORT}/v1/models"
served_name = "${SERVED_NAME}"
deadline = time.time() + 600
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8")
            obj = json.loads(body)
            model_ids = {item.get("id") for item in obj.get("data", [])}
            if served_name in model_ids:
                print(body[:500])
                raise SystemExit(0)
            last_error = f"served model {served_name!r} not in {sorted(model_ids)}"
    except Exception as exc:
        last_error = exc
    time.sleep(5)
raise SystemExit(f"server did not become healthy: {last_error}")
PY

for C in $CONCURRENCY_LIST; do
  echo "Running concurrency=$C"
  python scripts/bench_vlm_async.py \
    --base_url http://127.0.0.1:"$PORT"/v1 \
    --model "$SERVED_NAME" \
    --data "$DATA" \
    --num_prompts "$NUM_PROMPTS" \
    --concurrency "$C" \
    --max_tokens "$MAX_TOKENS" \
    --out results/"${SLURM_JOB_ID}_${MODEL_KIND}_c${C}.jsonl"
  sleep 5
done

python scripts/collect_results.py
echo "Done."
