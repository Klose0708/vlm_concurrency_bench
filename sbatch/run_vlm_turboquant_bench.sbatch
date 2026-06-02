#!/usr/bin/env bash
#SBATCH -J tq_vlm_bench
#SBATCH -o /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j_vlm_turboquant.out
#SBATCH -e /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j_vlm_turboquant.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=08:00:00

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

MODEL=${MODEL:-/data/private/"$USER"/workspace/models/Qwen2.5-VL-3B-Instruct}
DATA=${DATA:-data/mmstar/mmstar_requests.jsonl}
DATA_LIMIT=${DATA_LIMIT:-1500}
CONCURRENCY=${CONCURRENCY:-"1,2,4"}
NUM_REQUESTS=${NUM_REQUESTS:-24}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
TQ_BITS=${TQ_BITS:-4}
RESIDUAL_LEN=${RESIDUAL_LEN:-128}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
OUTPUT_DIR=${OUTPUT_DIR:-results/vlm_turboquant_${SLURM_JOB_ID}}
GPU_LOG=logs/"${SLURM_JOB_ID}_vlm_turboquant_gpu.csv"

if [[ ! -d "$MODEL" ]]; then
  echo "Model path does not exist: $MODEL"
  exit 2
fi

mkdir -p data/mmstar results logs
if [[ ! -f "$DATA" ]]; then
  python scripts/prepare_mmstar.py --out_dir "$(dirname "$DATA")" --limit "$DATA_LIMIT"
fi

chmod +x scripts/gpu_monitor.sh
bash scripts/gpu_monitor.sh "$GPU_LOG" 1 &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Running VLM TurboQuant benchmark on $(hostname)"
echo "MODEL=$MODEL"
echo "DATA=$DATA"
echo "CONCURRENCY=$CONCURRENCY NUM_REQUESTS=$NUM_REQUESTS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"

python scripts/bench_vlm_turboquant_concurrency.py \
  --model "$MODEL" \
  --data "$DATA" \
  --output_dir "$OUTPUT_DIR" \
  --modes baseline,turboquant \
  --concurrency "$CONCURRENCY" \
  --num_requests "$NUM_REQUESTS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --tq_bits "$TQ_BITS" \
  --residual_len "$RESIDUAL_LEN" \
  --torch_dtype "$TORCH_DTYPE"

echo "GPU log: $GPU_LOG"
echo "Output dir: $OUTPUT_DIR"
echo "Done."
