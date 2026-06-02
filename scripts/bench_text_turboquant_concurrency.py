#!/usr/bin/env python3
"""Concurrent text-generation benchmark for TurboQuant KV cache.

This script intentionally avoids upgrading the existing vLLM/VLM environment.
It adapts the open-source HuggingFace TurboQuant implementation under
``third_party/turboquant-back2matching`` to the Transformers 4.51 Cache API.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache


REPO_ROOT = Path(__file__).resolve().parents[1]
TURBOQUANT_SRC = REPO_ROOT / "third_party" / "turboquant-back2matching"
if not (TURBOQUANT_SRC / "turboquant" / "core.py").exists():
    TURBOQUANT_SRC = REPO_ROOT / "vendor" / "turboquant-back2matching"

spec = importlib.util.spec_from_file_location("turboquant_core", TURBOQUANT_SRC / "turboquant" / "core.py")
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to load TurboQuant core from {TURBOQUANT_SRC}")
turboquant_core = importlib.util.module_from_spec(spec)
sys.modules["turboquant_core"] = turboquant_core
spec.loader.exec_module(turboquant_core)
TurboQuantMSE = turboquant_core.TurboQuantMSE
pack_uint4 = turboquant_core.pack_uint4
unpack_uint4 = turboquant_core.unpack_uint4


def mb(num_bytes: int | float) -> float:
    return float(num_bytes) / 1024 / 1024


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.nelement() * tensor.element_size()


class TurboQuantLayer:
    """One layer of compressed KV state plus a recent FP16 residual window."""

    def __init__(self, bits: int, residual_len: int, layer_idx: int):
        self.bits = bits
        self.residual_len = residual_len
        self.layer_idx = layer_idx
        self.key_quantizer: TurboQuantMSE | None = None
        self.value_quantizer: TurboQuantMSE | None = None
        self.key_indices: torch.Tensor | None = None
        self.key_norms: torch.Tensor | None = None
        self.value_indices: torch.Tensor | None = None
        self.value_norms: torch.Tensor | None = None
        self.residual_keys: torch.Tensor | None = None
        self.residual_values: torch.Tensor | None = None
        self.head_dim: int | None = None
        self.total_len = 0
        self.dtype: torch.dtype | None = None

    def _ensure_quantizers(self, head_dim: int, device: torch.device) -> None:
        if self.key_quantizer is not None:
            return
        device_name = str(device)
        self.head_dim = head_dim
        self.key_quantizer = TurboQuantMSE(
            dim=head_dim, bits=self.bits, device=device_name, seed=42 + self.layer_idx * 17
        )
        self.value_quantizer = TurboQuantMSE(
            dim=head_dim, bits=self.bits, device=device_name, seed=42 + self.layer_idx * 17 + 1
        )

    def _pack(self, indices: torch.Tensor) -> torch.Tensor:
        if self.bits == 4:
            return pack_uint4(indices)
        return indices.to(torch.uint8)

    def _unpack(self, indices: torch.Tensor) -> torch.Tensor:
        if self.bits == 4:
            assert self.head_dim is not None
            return unpack_uint4(indices, self.head_dim)
        return indices

    def _quantize_append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        assert self.key_quantizer is not None
        assert self.value_quantizer is not None
        assert self.head_dim is not None
        k_idx, k_norms = self.key_quantizer.quantize(keys.reshape(-1, self.head_dim))
        v_idx, v_norms = self.value_quantizer.quantize(values.reshape(-1, self.head_dim))

        packed_k = self._pack(k_idx.reshape(keys.shape)).contiguous()
        packed_v = self._pack(v_idx.reshape(values.shape)).contiguous()
        k_norms = k_norms.reshape(keys.shape[:-1] + (1,)).contiguous()
        v_norms = v_norms.reshape(values.shape[:-1] + (1,)).contiguous()

        self.key_indices = (
            packed_k
            if self.key_indices is None
            else torch.cat([self.key_indices, packed_k], dim=-2)
        )
        self.value_indices = (
            packed_v
            if self.value_indices is None
            else torch.cat([self.value_indices, packed_v], dim=-2)
        )
        self.key_norms = (
            k_norms if self.key_norms is None else torch.cat([self.key_norms, k_norms], dim=-2)
        )
        self.value_norms = (
            v_norms
            if self.value_norms is None
            else torch.cat([self.value_norms, v_norms], dim=-2)
        )

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.dtype = keys.dtype
        self._ensure_quantizers(keys.shape[-1], keys.device)
        self.total_len += keys.shape[-2]

        if self.residual_keys is None:
            self.residual_keys = keys
            self.residual_values = values
        else:
            self.residual_keys = torch.cat([self.residual_keys, keys], dim=-2)
            self.residual_values = torch.cat([self.residual_values, values], dim=-2)

        if self.residual_keys.shape[-2] > self.residual_len:
            overflow = self.residual_keys.shape[-2] - self.residual_len
            self._quantize_append(
                self.residual_keys[..., :overflow, :],
                self.residual_values[..., :overflow, :],
            )
            self.residual_keys = self.residual_keys[..., overflow:, :].contiguous()
            self.residual_values = self.residual_values[..., overflow:, :].contiguous()

        return self.materialize()

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.residual_keys is not None
        assert self.residual_values is not None
        if self.key_indices is None:
            return self.residual_keys, self.residual_values

        assert self.key_quantizer is not None
        assert self.value_quantizer is not None
        assert self.head_dim is not None
        assert self.key_norms is not None
        assert self.value_norms is not None
        assert self.value_indices is not None

        key_idx = self._unpack(self.key_indices)
        value_idx = self._unpack(self.value_indices)
        key_deq = self.key_quantizer.dequantize(
            key_idx.reshape(-1, self.head_dim), self.key_norms.reshape(-1, 1)
        ).reshape(key_idx.shape)
        value_deq = self.value_quantizer.dequantize(
            value_idx.reshape(-1, self.head_dim), self.value_norms.reshape(-1, 1)
        ).reshape(value_idx.shape)

        key_deq = key_deq.to(dtype=self.dtype)
        value_deq = value_deq.to(dtype=self.dtype)
        return (
            torch.cat([key_deq, self.residual_keys], dim=-2),
            torch.cat([value_deq, self.residual_values], dim=-2),
        )

    def memory_bytes(self) -> int:
        return (
            tensor_bytes(self.key_indices)
            + tensor_bytes(self.key_norms)
            + tensor_bytes(self.value_indices)
            + tensor_bytes(self.value_norms)
            + tensor_bytes(self.residual_keys)
            + tensor_bytes(self.residual_values)
        )

    def fp16_equivalent_bytes(self) -> int:
        if self.head_dim is None or self.residual_keys is None:
            return 0
        batch = self.residual_keys.shape[0]
        heads = self.residual_keys.shape[1]
        return batch * heads * self.total_len * self.head_dim * 2 * 2


class CompatTurboQuantCache(Cache):
    """Transformers 4.51-compatible TurboQuant cache."""

    def __init__(self, bits: int = 4, residual_len: int = 128):
        super().__init__()
        self.bits = bits
        self.residual_len = residual_len
        self.layers: dict[int, TurboQuantLayer] = {}
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        if layer_idx not in self.layers:
            self.layers[layer_idx] = TurboQuantLayer(
                bits=self.bits, residual_len=self.residual_len, layer_idx=layer_idx
            )
        return self.layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        layer = self.layers.get(layer_idx)
        return 0 if layer is None else layer.total_len

    def get_max_cache_shape(self) -> None:
        return None

    def memory_usage_bytes(self) -> dict[str, float]:
        compressed = sum(layer.memory_bytes() for layer in self.layers.values())
        fp16_equiv = sum(layer.fp16_equivalent_bytes() for layer in self.layers.values())
        return {
            "compressed_bytes": compressed,
            "fp16_equivalent_bytes": fp16_equiv,
            "savings_ratio": fp16_equiv / max(compressed, 1),
        }


@dataclass
class RequestResult:
    mode: str
    execution_mode: str
    concurrency: int
    request_id: int
    ok: bool
    error: str
    latency_s: float
    ttft_s: float
    decode_s: float
    prompt_tokens: int
    output_tokens: int
    tokens_per_s: float
    peak_allocated_mb: float
    cache_compressed_mb: float
    cache_fp16_equiv_mb: float
    cache_savings_ratio: float
    output_preview: str


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def make_prompt(tokenizer: Any, target_tokens: int, request_id: int) -> str:
    filler = (
        "In a controlled benchmark, the system records latency, memory, and output "
        "quality for long-context language-model inference. "
    )
    question = (
        f"\n\nRequest {request_id}: summarize the benchmark in exactly one concise "
        "Chinese sentence and include the word OK."
    )
    filler_tokens = max(1, len(tokenizer.encode(filler)))
    question_tokens = len(tokenizer.encode(question))
    repeats = max(1, (target_tokens - question_tokens) // filler_tokens)
    return filler * repeats + question


def make_text_dataset_prompts(
    *,
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    text_field: str,
    target_tokens: int,
    num_prompts: int,
    max_rows: int,
) -> list[str]:
    """Build benchmark prompts from a public text dataset.

    The prompt body is real dataset text. A short instruction is appended so the
    model has a deterministic continuation task while preserving the requested
    context length.
    """

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    texts: list[str] = []
    for index, item in enumerate(dataset):
        if max_rows > 0 and index >= max_rows:
            break
        text = str(item.get(text_field, "")).strip()
        if text and not (text.startswith("=") and text.endswith("=")):
            texts.append(text)

    if not texts:
        raise RuntimeError(f"no usable text found in {dataset_name}/{dataset_config}:{dataset_split}")

    instruction = (
        "\n\nContinue the passage above in exactly one concise English sentence."
    )
    instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
    context_budget = max(16, target_tokens - len(instruction_tokens))
    corpus = "\n\n".join(texts)
    token_ids = tokenizer.encode(corpus, add_special_tokens=False)
    if len(token_ids) < context_budget:
        raise RuntimeError(
            f"dataset text is too short after tokenization: {len(token_ids)} < {context_budget}"
        )

    prompts: list[str] = []
    max_start = max(1, len(token_ids) - context_budget)
    stride = max(1, context_budget // 2)
    for request_id in range(num_prompts):
        start = (request_id * stride) % max_start
        window = token_ids[start : start + context_budget]
        text = tokenizer.decode(window, skip_special_tokens=True)
        prompts.append(text + instruction)
    return prompts


def reset_cuda_stats() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cache_memory_stats(cache: Any) -> dict[str, float]:
    if hasattr(cache, "memory_usage_bytes"):
        return cache.memory_usage_bytes()

    key_cache = getattr(cache, "key_cache", [])
    value_cache = getattr(cache, "value_cache", [])
    total = 0
    for tensor in list(key_cache) + list(value_cache):
        total += tensor_bytes(tensor)
    return {
        "compressed_bytes": total,
        "fp16_equivalent_bytes": total,
        "savings_ratio": 1.0,
    }


def run_one(
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    prompt: str,
    request_id: int,
    concurrency: int,
    max_new_tokens: int,
    tq_bits: int,
    residual_len: int,
    model_lock: Lock,
) -> RequestResult:
    started = time.perf_counter()
    try:
        with model_lock:
            reset_cuda_stats()
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_tokens = int(inputs["input_ids"].shape[1])
            cache: Cache
            if mode == "turboquant":
                cache = CompatTurboQuantCache(bits=tq_bits, residual_len=residual_len)
            else:
                cache = DynamicCache()

            prefill_started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=True, past_key_values=cache)
            ttft_s = time.perf_counter() - prefill_started

            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [int(next_token.item())]

            decode_started = time.perf_counter()
            with torch.inference_mode():
                for _ in range(max_new_tokens - 1):
                    outputs = model(input_ids=next_token, use_cache=True, past_key_values=past)
                    past = outputs.past_key_values
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token_id = int(next_token.item())
                    generated.append(token_id)
                    if token_id == tokenizer.eos_token_id:
                        break
            decode_s = time.perf_counter() - decode_started

            peak_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
                if torch.cuda.is_available()
                else 0.0
            )
            cache_stats = cache_memory_stats(past)
            output = tokenizer.decode(generated, skip_special_tokens=True)

        latency_s = time.perf_counter() - started
        return RequestResult(
            mode=mode,
            execution_mode="serial",
            concurrency=concurrency,
            request_id=request_id,
            ok=True,
            error="",
            latency_s=latency_s,
            ttft_s=ttft_s,
            decode_s=decode_s,
            prompt_tokens=prompt_tokens,
            output_tokens=len(generated),
            tokens_per_s=len(generated) / max(decode_s, 1e-9),
            peak_allocated_mb=peak_mb,
            cache_compressed_mb=mb(cache_stats["compressed_bytes"]),
            cache_fp16_equiv_mb=mb(cache_stats["fp16_equivalent_bytes"]),
            cache_savings_ratio=float(cache_stats["savings_ratio"]),
            output_preview=output[:120],
        )
    except Exception as exc:  # noqa: BLE001 - experiment should record failures
        return RequestResult(
            mode=mode,
            execution_mode="serial",
            concurrency=concurrency,
            request_id=request_id,
            ok=False,
            error=repr(exc),
            latency_s=time.perf_counter() - started,
            ttft_s=0.0,
            decode_s=0.0,
            prompt_tokens=0,
            output_tokens=0,
            tokens_per_s=0.0,
            peak_allocated_mb=0.0,
            cache_compressed_mb=0.0,
            cache_fp16_equiv_mb=0.0,
            cache_savings_ratio=0.0,
            output_preview="",
        )


async def run_concurrency(
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    concurrency: int,
    num_requests: int,
    prompt_tokens: int,
    max_new_tokens: int,
    tq_bits: int,
    residual_len: int,
    model_lock: Lock,
    prompts: list[str] | None = None,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def task(request_id: int) -> RequestResult:
        prompt = (
            prompts[request_id % len(prompts)]
            if prompts
            else make_prompt(tokenizer, prompt_tokens, request_id)
        )
        async with semaphore:
            return await asyncio.to_thread(
                run_one,
                model=model,
                tokenizer=tokenizer,
                mode=mode,
                prompt=prompt,
                request_id=request_id,
                concurrency=concurrency,
                max_new_tokens=max_new_tokens,
                tq_bits=tq_bits,
                residual_len=residual_len,
                model_lock=model_lock,
            )

    return await asyncio.gather(*(task(i) for i in range(num_requests)))


def run_batch_wave(
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    prompts: list[str],
    request_ids: list[int],
    concurrency: int,
    max_new_tokens: int,
    tq_bits: int,
    residual_len: int,
) -> list[RequestResult]:
    started = time.perf_counter()
    batch_size = len(prompts)
    try:
        reset_cuda_stats()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_counts = [int(mask.sum().item()) for mask in inputs["attention_mask"]]
        cache: Cache
        if mode == "turboquant":
            cache = CompatTurboQuantCache(bits=tq_bits, residual_len=residual_len)
        else:
            cache = DynamicCache()

        prefill_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True, past_key_values=cache)
        ttft_s = time.perf_counter() - prefill_started

        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]

        decode_started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(max_new_tokens - 1):
                outputs = model(input_ids=next_token, use_cache=True, past_key_values=past)
                past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)
        decode_s = time.perf_counter() - decode_started

        generated_tensor = torch.cat(generated, dim=1)
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        )
        cache_stats = cache_memory_stats(past)
        compressed_mb_per_request = mb(cache_stats["compressed_bytes"]) / max(batch_size, 1)
        fp16_mb_per_request = mb(cache_stats["fp16_equivalent_bytes"]) / max(batch_size, 1)
        latency_s = time.perf_counter() - started

        results = []
        for row, request_id in enumerate(request_ids):
            output_ids = generated_tensor[row].tolist()
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append(
                RequestResult(
                    mode=mode,
                    execution_mode="batch",
                    concurrency=concurrency,
                    request_id=request_id,
                    ok=True,
                    error="",
                    latency_s=latency_s,
                    ttft_s=ttft_s,
                    decode_s=decode_s,
                    prompt_tokens=prompt_token_counts[row],
                    output_tokens=len(output_ids),
                    tokens_per_s=(len(output_ids) * batch_size) / max(decode_s, 1e-9),
                    peak_allocated_mb=peak_mb,
                    cache_compressed_mb=compressed_mb_per_request,
                    cache_fp16_equiv_mb=fp16_mb_per_request,
                    cache_savings_ratio=float(cache_stats["savings_ratio"]),
                    output_preview=output[:120],
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001 - experiment should record failures
        latency_s = time.perf_counter() - started
        return [
            RequestResult(
                mode=mode,
                execution_mode="batch",
                concurrency=concurrency,
                request_id=request_id,
                ok=False,
                error=repr(exc),
                latency_s=latency_s,
                ttft_s=0.0,
                decode_s=0.0,
                prompt_tokens=0,
                output_tokens=0,
                tokens_per_s=0.0,
                peak_allocated_mb=0.0,
                cache_compressed_mb=0.0,
                cache_fp16_equiv_mb=0.0,
                cache_savings_ratio=0.0,
                output_preview="",
            )
            for request_id in request_ids
        ]


def run_batch_concurrency(
    *,
    model: Any,
    tokenizer: Any,
    mode: str,
    concurrency: int,
    num_requests: int,
    prompt_tokens: int,
    max_new_tokens: int,
    tq_bits: int,
    residual_len: int,
    dataset_prompts: list[str] | None = None,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    request_id = 0
    while request_id < num_requests:
        wave_ids = list(range(request_id, min(request_id + concurrency, num_requests)))
        prompts = [
            dataset_prompts[rid % len(dataset_prompts)]
            if dataset_prompts
            else make_prompt(tokenizer, prompt_tokens, rid)
            for rid in wave_ids
        ]
        results.extend(
            run_batch_wave(
                model=model,
                tokenizer=tokenizer,
                mode=mode,
                prompts=prompts,
                request_ids=wave_ids,
                concurrency=concurrency,
                max_new_tokens=max_new_tokens,
                tq_bits=tq_bits,
                residual_len=residual_len,
            )
        )
        request_id += concurrency
    return results


def summarize(results: list[RequestResult], elapsed_s: float) -> dict[str, Any]:
    ok_results = [r for r in results if r.ok]
    latencies = [r.latency_s for r in ok_results]
    ttfts = [r.ttft_s for r in ok_results]
    tokens = sum(r.output_tokens for r in ok_results)
    return {
        "mode": results[0].mode if results else "",
        "execution_mode": results[0].execution_mode if results else "",
        "concurrency": results[0].concurrency if results else 0,
        "num_requests": len(results),
        "ok_requests": len(ok_results),
        "error_rate": 1.0 - len(ok_results) / max(len(results), 1),
        "elapsed_s": elapsed_s,
        "requests_per_s": len(ok_results) / max(elapsed_s, 1e-9),
        "output_tokens_per_s": tokens / max(elapsed_s, 1e-9),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "ttft_mean_s": statistics.mean(ttfts) if ttfts else 0.0,
        "ttft_p95_s": percentile(ttfts, 95),
        "decode_tokens_per_s_mean": statistics.mean([r.tokens_per_s for r in ok_results])
        if ok_results
        else 0.0,
        "peak_allocated_mb_max": max([r.peak_allocated_mb for r in ok_results], default=0.0),
        "cache_compressed_mb_mean": statistics.mean([r.cache_compressed_mb for r in ok_results])
        if ok_results
        else 0.0,
        "cache_fp16_equiv_mb_mean": statistics.mean([r.cache_fp16_equiv_mb for r in ok_results])
        if ok_results
        else 0.0,
        "cache_savings_ratio_mean": statistics.mean([r.cache_savings_ratio for r in ok_results])
        if ok_results
        else 0.0,
        "sample_output": ok_results[0].output_preview if ok_results else "",
        "first_error": next((r.error for r in results if not r.ok), ""),
    }


def write_outputs(
    output_dir: Path, all_results: list[RequestResult], summaries: list[dict[str, Any]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "text_turboquant_requests.jsonl").open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    with (output_dir / "text_turboquant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    if summaries:
        with (output_dir / "text_turboquant_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=f"/data/private/{os.environ.get('USER', 's202510003')}/workspace/models/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--output_dir", default="results/text_turboquant")
    parser.add_argument("--modes", default="baseline,turboquant")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--num_requests", type=int, default=8)
    parser.add_argument("--prompt_tokens", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--tq_bits", type=int, default=4)
    parser.add_argument("--residual_len", type=int, default=128)
    parser.add_argument(
        "--dataset_name",
        default="",
        help="Optional HuggingFace text dataset name, e.g. Salesforce/wikitext.",
    )
    parser.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset_split", default="validation")
    parser.add_argument("--dataset_text_field", default="text")
    parser.add_argument(
        "--dataset_max_rows",
        type=int,
        default=0,
        help="Maximum dataset rows to scan when building prompts; 0 means all rows.",
    )
    parser.add_argument(
        "--execution_mode",
        choices=["serial", "batch"],
        default="serial",
        help="serial uses threaded request simulation; batch runs each concurrency wave as one model batch.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    model_lock = Lock()
    all_results: list[RequestResult] = []
    summaries: list[dict[str, Any]] = []
    dataset_prompts = (
        make_text_dataset_prompts(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_field=args.dataset_text_field,
            target_tokens=args.prompt_tokens,
            num_prompts=args.num_requests,
            max_rows=args.dataset_max_rows,
        )
        if args.dataset_name
        else None
    )

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        for concurrency in [int(c) for c in args.concurrency.split(",") if c.strip()]:
            print(f"Running mode={mode}, concurrency={concurrency}", flush=True)
            started = time.perf_counter()
            if args.execution_mode == "batch":
                results = run_batch_concurrency(
                    model=model,
                    tokenizer=tokenizer,
                    mode=mode,
                    concurrency=concurrency,
                    num_requests=args.num_requests,
                    prompt_tokens=args.prompt_tokens,
                    max_new_tokens=args.max_new_tokens,
                    tq_bits=args.tq_bits,
                    residual_len=args.residual_len,
                    dataset_prompts=dataset_prompts,
                )
            else:
                results = await run_concurrency(
                    model=model,
                    tokenizer=tokenizer,
                    mode=mode,
                    concurrency=concurrency,
                    num_requests=args.num_requests,
                    prompt_tokens=args.prompt_tokens,
                    max_new_tokens=args.max_new_tokens,
                    tq_bits=args.tq_bits,
                    residual_len=args.residual_len,
                    model_lock=model_lock,
                    prompts=dataset_prompts,
                )
            elapsed_s = time.perf_counter() - started
            all_results.extend(results)
            summary = summarize(results, elapsed_s)
            summary.update(
                {
                    "prompt_source": args.dataset_name or "synthetic",
                    "dataset_config": args.dataset_config if args.dataset_name else "",
                    "dataset_split": args.dataset_split if args.dataset_name else "",
                    "target_prompt_tokens": args.prompt_tokens,
                }
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    write_outputs(Path(args.output_dir), all_results, summaries)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
