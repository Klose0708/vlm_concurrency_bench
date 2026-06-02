# VLM Concurrency Bench

This repository contains the core experiment code, Slurm launch scripts, reports, and small/medium dataset artifacts for VLM and TurboQuant concurrency benchmarks.

## Contents

- `scripts/`: benchmark clients, dataset preparation scripts, result collection, and GPU monitoring.
- `sbatch/`: Slurm job scripts for VLM serving benchmarks and TurboQuant benchmarks.
- `reports/`: Chinese technical reports and experiment summaries.
- `docs/`: research notes.
- `data/`: prepared request files and associated lightweight benchmark data used by the experiments.
- `vendor/turboquant-back2matching/turboquant/`: vendored TurboQuant core file used by the HuggingFace cache adapter when the original `third_party/` clone is absent.

Runtime outputs such as `logs/` and `results/` are intentionally ignored by Git. Model weights and HuggingFace caches are also excluded and should be prepared on the target cluster.

## Key Experiments

### VLM Serving Baselines

Run Qwen2.5-VL with vLLM BF16, FP8 KV cache, or AWQ:

```bash
sbatch --export=ALL,MODEL_KIND=bf16,DATA=data/mmstar_full/mmstar_requests.jsonl,NUM_PROMPTS=1500,MAX_TOKENS=16,CONCURRENCY_LIST="1 2 4 8 16 32" sbatch/run_vlm_bench.sbatch
```

### VLM TurboQuant vs DynamicCache

Run Qwen2.5-VL through HuggingFace with ordinary `DynamicCache` and TurboQuant KV cache:

```bash
CONCURRENCY="1,2,4,8" NUM_REQUESTS=128 MAX_NEW_TOKENS=16 OUTPUT_DIR=results/vlm_turboquant_formal_mmstar128_leftpad_c1248 \
  sbatch --export=ALL sbatch/run_vlm_turboquant_bench.sbatch
```

### Text TurboQuant on WikiText-103

Run text-only TurboQuant experiments with the public `Salesforce/wikitext` dataset:

```bash
DATASET_NAME=Salesforce/wikitext DATASET_CONFIG=wikitext-103-raw-v1 DATASET_SPLIT=validation \
CONCURRENCY="1,2,4,8" NUM_REQUESTS=32 PROMPT_TOKENS=2048 EXECUTION_MODE=batch \
OUTPUT_DIR=results/text_turboquant_wikitext103_ctx2048_c1248 \
  sbatch --export=ALL sbatch/run_text_turboquant_bench.sbatch
```

All GPU experiments should be submitted to Slurm compute nodes. Do not run formal inference experiments directly on the login/admin node.
