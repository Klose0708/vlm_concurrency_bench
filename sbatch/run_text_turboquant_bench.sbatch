#!/usr/bin/env bash
#SBATCH -J tq_text_bench
#SBATCH -o /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j_text_turboquant.out
#SBATCH -e /data/private/%u/workspace/projects/vlm_concurrency_bench/logs/%j_text_turboquant.err
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00

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

MODEL=${MODEL:-/data/private/"$USER"/workspace/models/Qwen2.5-0.5B-Instruct}
CONCURRENCY=${CONCURRENCY:-"1,2,4"}
NUM_REQUESTS=${NUM_REQUESTS:-8}
PROMPT_TOKENS=${PROMPT_TOKENS:-768}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
TQ_BITS=${TQ_BITS:-4}
RESIDUAL_LEN=${RESIDUAL_LEN:-128}
EXECUTION_MODE=${EXECUTION_MODE:-serial}
DATASET_NAME=${DATASET_NAME:-}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-103-raw-v1}
DATASET_SPLIT=${DATASET_SPLIT:-validation}
DATASET_TEXT_FIELD=${DATASET_TEXT_FIELD:-text}
DATASET_MAX_ROWS=${DATASET_MAX_ROWS:-0}
OUTPUT_DIR=${OUTPUT_DIR:-results/text_turboquant_${SLURM_JOB_ID}}
GPU_LOG=logs/"${SLURM_JOB_ID}_text_turboquant_gpu.csv"

if [[ ! -d "$MODEL" ]]; then
  echo "Model path does not exist: $MODEL"
  exit 2
fi

chmod +x scripts/gpu_monitor.sh
bash scripts/gpu_monitor.sh "$GPU_LOG" 1 &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Running text TurboQuant benchmark on $(hostname)"
echo "MODEL=$MODEL"
echo "CONCURRENCY=$CONCURRENCY NUM_REQUESTS=$NUM_REQUESTS PROMPT_TOKENS=$PROMPT_TOKENS MAX_NEW_TOKENS=$MAX_NEW_TOKENS EXECUTION_MODE=$EXECUTION_MODE"
echo "DATASET_NAME=${DATASET_NAME:-synthetic} DATASET_CONFIG=$DATASET_CONFIG DATASET_SPLIT=$DATASET_SPLIT"

DATASET_ARGS=()
if [[ -n "$DATASET_NAME" ]]; then
  DATASET_ARGS=(
    --dataset_name "$DATASET_NAME"
    --dataset_config "$DATASET_CONFIG"
    --dataset_split "$DATASET_SPLIT"
    --dataset_text_field "$DATASET_TEXT_FIELD"
    --dataset_max_rows "$DATASET_MAX_ROWS"
  )
fi

python scripts/bench_text_turboquant_concurrency.py \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --modes baseline,turboquant \
  --concurrency "$CONCURRENCY" \
  --num_requests "$NUM_REQUESTS" \
  --prompt_tokens "$PROMPT_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --tq_bits "$TQ_BITS" \
  --residual_len "$RESIDUAL_LEN" \
  --execution_mode "$EXECUTION_MODE" \
  "${DATASET_ARGS[@]}"

echo "GPU log: $GPU_LOG"
echo "Output dir: $OUTPUT_DIR"
echo "Done."
