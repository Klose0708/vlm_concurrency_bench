# VLM Concurrency Quantization Experiment Report

## Scope

- Dataset: MMStar full validation set, 1500 image-question samples, prepared from ModelScope `evalscope/MMStar` parquet.
- Model: Qwen2.5-VL-3B-Instruct on vLLM 0.8.5.post1.
- Configurations: BF16, BF16 + FP8 KV cache, AWQ.
- AWQ used `--enforce-eager` because AWQ Marlin with torch.compile produced CUDA illegal memory access during warmup.
- Output cap: 16 tokens. Prompt requires a single A/B/C/D answer.

## Key Findings

- BF16 was the strongest stable throughput baseline in the full run: throughput kept increasing up to concurrency 32 with zero errors.
- FP8 KV cache was stable but slower than BF16 on this short-output VLM workload; KV cache is not yet the dominant bottleneck at these prompt/output lengths.
- AWQ was stable for concurrency 1, 2, and 4 in the full run. Previous 500-sample probing found high-concurrency instability at 8 and above.
- Accuracy stayed close across stable configurations: BF16 about 58.5%, FP8 KV about 58.5-59.3%, AWQ about 58.9% on full MMStar.

AWQ 500-sample high-concurrency probe showed instability: c=8 error_rate=0.53, c=16 error_rate=1.00, c=32 error_rate=1.00. Server log for job 9120 records vLLM EngineCore death caused by `CUDA error: uncorrectable ECC error encountered`.

## Best Zero-Error Point Per Configuration

| config | best_zero_error_concurrency | requests_per_s | latency_p95_s | accuracy |
| --- | --- | --- | --- | --- |
| AWQ | 4 | 12.6400 | 0.6249 | 0.5347 |
| BF16 | 32 | 25.5822 | 2.2261 | 0.5453 |
| BF16 + FP8 KV | 32 | 14.5211 | 3.8216 | 0.5473 |

## Full Formal Summary

| config | concurrency | requests_per_s | latency_p95_s | ttft_p95_s | accuracy | error_rate |
| --- | --- | --- | --- | --- | --- | --- |
| AWQ | 1 | 5.3212 | 0.2555 | 0.2135 | 0.5347 | 0.0000 |
| AWQ | 2 | 8.3090 | 0.3484 | 0.2849 | 0.5347 | 0.0000 |
| AWQ | 4 | 12.6400 | 0.6249 | 0.4840 | 0.5347 | 0.0000 |
| BF16 | 1 | 6.4846 | 0.2420 | 0.2286 | 0.5453 | 0.0000 |
| BF16 | 2 | 9.2016 | 0.3311 | 0.3010 | 0.5460 | 0.0000 |
| BF16 | 4 | 13.9090 | 0.6100 | 0.4991 | 0.5480 | 0.0000 |
| BF16 | 8 | 18.5592 | 1.0177 | 0.9217 | 0.5467 | 0.0000 |
| BF16 | 16 | 23.5152 | 1.4679 | 1.3003 | 0.5480 | 0.0000 |
| BF16 | 32 | 25.5822 | 2.2261 | 1.9962 | 0.5453 | 0.0000 |
| BF16 + FP8 KV | 1 | 6.4780 | 0.2095 | 0.1947 | 0.5473 | 0.0000 |
| BF16 + FP8 KV | 2 | 10.2878 | 0.3455 | 0.3283 | 0.5487 | 0.0000 |
| BF16 + FP8 KV | 4 | 12.4116 | 0.6473 | 0.6306 | 0.5520 | 0.0000 |
| BF16 + FP8 KV | 8 | 13.8126 | 1.2441 | 1.2181 | 0.5493 | 0.0000 |
| BF16 + FP8 KV | 16 | 14.0384 | 2.2272 | 2.0315 | 0.5487 | 0.0000 |
| BF16 + FP8 KV | 32 | 14.5211 | 3.8216 | 3.0704 | 0.5473 | 0.0000 |

## Artifacts

- Formal CSV: `reports/formal_summary_1500.csv`
- AWQ failure probe CSV: `reports/awq_instability_probe_500.csv`
- Raw results: `results/9126_*`, `results/9127_*`, `results/9128_*`
- GPU logs: `logs/9126_*_gpu.csv`, `logs/9127_*_gpu.csv`, `logs/9128_*_gpu.csv`
