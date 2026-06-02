# TurboQuant Research Notes

## Core Paper

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate, arXiv:2504.19874.
- Google Research describes TurboQuant as an ICLR 2026 work for online vector quantization, KV-cache compression, and vector search.
- The algorithm is data-oblivious and online: it does not require calibration data or training codebooks before vectors arrive.
- The MSE path applies a random rotation, then uses near-optimal scalar quantizers per coordinate after the rotated coordinates become concentrated.
- The inner-product path first applies the MSE quantizer, then applies 1-bit QJL to the residual, producing an unbiased inner-product estimator.
- Reported KV-cache results emphasize roughly 3 to 4 bits per channel, quality neutrality around 3.5 bits per channel, and marginal degradation around 2.5 bits per channel.

## Closely Related Work

- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead, arXiv:2406.03482 / AAAI 2025. It uses a Johnson-Lindenstrauss transform plus sign-bit quantization and avoids storing scale/zero-point metadata.
- PolarQuant: Quantizing KV Caches with Polar Transformation, arXiv:2502.02617 / Google Research 2025. It uses random preconditioning and polar coordinates to reduce normalization overhead for KV-cache quantization.
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache, ICML 2024. It quantizes key cache per-channel and value cache per-token, with reported peak-memory and throughput benefits.
- KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization, NeurIPS 2024. It studies low-bit KV-cache quantization with outlier-aware and non-uniform methods.
- vLLM FP8 KV cache is the practical baseline available in the current serving stack; it is not TurboQuant, but it tests the same service bottleneck: KV-cache memory pressure under larger batches, longer contexts, or higher concurrency.

## Experiment Implication

TurboQuant is not currently a one-flag VLM serving mode in vLLM. For this project, the correct first step is to run deployable baselines:

- BF16 Qwen2.5-VL-3B-Instruct.
- AWQ weight-quantized Qwen2.5-VL-3B-Instruct if the AWQ checkpoint exists.
- BF16 plus vLLM FP8 KV cache if supported by the installed vLLM, CUDA, and GPU backend.
- AWQ plus FP8 KV cache after both AWQ and FP8 KV work independently.

The main TurboQuant-aligned measurements are high-concurrency memory usage, TTFT growth, throughput saturation, error/OOM rate, and whether a KV-cache compression baseline preserves MMStar answer accuracy.
