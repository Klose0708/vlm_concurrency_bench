#!/usr/bin/env python3
"""Batch-concurrency benchmark for Qwen2.5-VL with TurboQuant KV cache.

The existing vLLM path is kept intact for service-style BF16/FP8/AWQ baselines.
This script runs Qwen2.5-VL through HuggingFace Transformers so the same
TurboQuant cache adapter used by the text benchmark can be applied to a real VLM.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers import AutoProcessor, DynamicCache

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - depends on cluster transformers version
    raise RuntimeError(
        "Qwen2_5_VLForConditionalGeneration is required. "
        "Use the existing vlm_bench_cu124 environment with transformers >= 4.51."
    ) from exc

from bench_text_turboquant_concurrency import (
    CompatTurboQuantCache,
    cache_memory_stats,
    mb,
    percentile,
)


# Some local Qwen2.5-VL checkpoints keep ``rope_scaling.rope_type=mrope``.
# In transformers 4.51 this is intended to use the default RoPE initializer,
# but older configs may bypass the conversion in the config class.
ROPE_INIT_FUNCTIONS.setdefault("mrope", ROPE_INIT_FUNCTIONS["default"])


@dataclass
class VLMRequestResult:
    mode: str
    concurrency: int
    request_id: int
    ok: bool
    error: str
    latency_s: float
    prompt_tokens: int
    output_tokens: int
    peak_allocated_mb: float
    cache_compressed_mb: float
    cache_fp16_equiv_mb: float
    cache_savings_ratio: float
    output_text: str
    pred: str
    gold: str
    correct: bool | None
    category: str
    l2_category: str


def parse_choice(text: str) -> str:
    normalized = (text or "").strip().upper()
    match = re.search(r"\b([ABCD])\b", normalized)
    if match:
        return match.group(1)
    return normalized[0] if normalized[:1] in {"A", "B", "C", "D"} else ""


def load_rows(path: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def reset_cuda_stats() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_batch_inputs(
    *,
    processor: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    images: list[Image.Image] = []
    for row in rows:
        image = Image.open(row["image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.append(image)

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items()}


def run_wave(
    *,
    model: Any,
    processor: Any,
    mode: str,
    rows: list[dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    tq_bits: int,
    residual_len: int,
) -> list[VLMRequestResult]:
    started = time.perf_counter()
    try:
        reset_cuda_stats()
        inputs = build_batch_inputs(processor=processor, rows=rows, device=model.device)
        prompt_token_counts = [int(mask.sum().item()) for mask in inputs["attention_mask"]]
        cache = (
            CompatTurboQuantCache(bits=tq_bits, residual_len=residual_len)
            if mode == "turboquant"
            else DynamicCache()
        )

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                use_cache=True,
                past_key_values=cache,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        latency_s = time.perf_counter() - started
        input_width = int(inputs["input_ids"].shape[1])
        generated_trimmed = [output_ids[input_width:] for output_ids in generated_ids]
        output_texts = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_token_counts = [int(ids.numel()) for ids in generated_trimmed]

        cache_stats = cache_memory_stats(cache)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
        batch_size = max(len(rows), 1)
        compressed_mb_per_request = mb(cache_stats["compressed_bytes"]) / batch_size
        fp16_mb_per_request = mb(cache_stats["fp16_equivalent_bytes"]) / batch_size
        results: list[VLMRequestResult] = []
        for row, prompt_tokens, output_tokens, output_text in zip(
            rows, prompt_token_counts, output_token_counts, output_texts, strict=True
        ):
            pred = parse_choice(output_text)
            gold = str(row.get("answer", "")).strip().upper()[:1]
            results.append(
                VLMRequestResult(
                    mode=mode,
                    concurrency=concurrency,
                    request_id=int(row.get("id", len(results))),
                    ok=True,
                    error="",
                    latency_s=latency_s,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    peak_allocated_mb=peak_mb,
                    cache_compressed_mb=compressed_mb_per_request,
                    cache_fp16_equiv_mb=fp16_mb_per_request,
                    cache_savings_ratio=float(cache_stats["savings_ratio"]),
                    output_text=output_text.strip(),
                    pred=pred,
                    gold=gold,
                    correct=(pred == gold) if gold else None,
                    category=str(row.get("category", "")),
                    l2_category=str(row.get("l2_category", "")),
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures
        latency_s = time.perf_counter() - started
        return [
            VLMRequestResult(
                mode=mode,
                concurrency=concurrency,
                request_id=int(row.get("id", index)),
                ok=False,
                error=repr(exc),
                latency_s=latency_s,
                prompt_tokens=0,
                output_tokens=0,
                peak_allocated_mb=0.0,
                cache_compressed_mb=0.0,
                cache_fp16_equiv_mb=0.0,
                cache_savings_ratio=0.0,
                output_text="",
                pred="",
                gold=str(row.get("answer", "")).strip().upper()[:1],
                correct=None,
                category=str(row.get("category", "")),
                l2_category=str(row.get("l2_category", "")),
            )
            for index, row in enumerate(rows)
        ]


def run_mode_concurrency(
    *,
    model: Any,
    processor: Any,
    mode: str,
    rows: list[dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    tq_bits: int,
    residual_len: int,
) -> tuple[list[VLMRequestResult], float]:
    all_results: list[VLMRequestResult] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), concurrency):
        wave = rows[offset : offset + concurrency]
        all_results.extend(
            run_wave(
                model=model,
                processor=processor,
                mode=mode,
                rows=wave,
                concurrency=concurrency,
                max_new_tokens=max_new_tokens,
                tq_bits=tq_bits,
                residual_len=residual_len,
            )
        )
    return all_results, time.perf_counter() - started


def summarize(results: list[VLMRequestResult], elapsed_s: float) -> dict[str, Any]:
    ok_results = [result for result in results if result.ok]
    latencies = [result.latency_s for result in ok_results]
    output_tokens = sum(result.output_tokens for result in ok_results)
    accuracies = [float(result.correct) for result in ok_results if result.correct is not None]
    return {
        "mode": results[0].mode if results else "",
        "concurrency": results[0].concurrency if results else 0,
        "num_requests": len(results),
        "ok_requests": len(ok_results),
        "error_rate": 1.0 - len(ok_results) / max(len(results), 1),
        "elapsed_s": elapsed_s,
        "requests_per_s": len(ok_results) / max(elapsed_s, 1e-9),
        "output_tokens_per_s": output_tokens / max(elapsed_s, 1e-9),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "accuracy": statistics.mean(accuracies) if accuracies else 0.0,
        "parse_success_rate": statistics.mean([bool(result.pred) for result in ok_results])
        if ok_results
        else 0.0,
        "peak_allocated_mb_max": max([result.peak_allocated_mb for result in ok_results], default=0.0),
        "cache_compressed_mb_mean": statistics.mean([result.cache_compressed_mb for result in ok_results])
        if ok_results
        else 0.0,
        "cache_fp16_equiv_mb_mean": statistics.mean([result.cache_fp16_equiv_mb for result in ok_results])
        if ok_results
        else 0.0,
        "cache_savings_ratio_mean": statistics.mean([result.cache_savings_ratio for result in ok_results])
        if ok_results
        else 0.0,
        "first_error": next((result.error for result in results if not result.ok), ""),
        "sample_output": ok_results[0].output_text[:120] if ok_results else "",
    }


def write_outputs(
    output_dir: Path,
    all_results: list[VLMRequestResult],
    summaries: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vlm_turboquant_requests.jsonl").open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    with (output_dir / "vlm_turboquant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    if summaries:
        with (output_dir / "vlm_turboquant_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=f"/data/private/{os.environ.get('USER', 's202510003')}/workspace/models/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument("--data", default="data/mmstar/mmstar_requests.jsonl")
    parser.add_argument("--output_dir", default="results/vlm_turboquant")
    parser.add_argument("--modes", default="baseline,turboquant")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--num_requests", type=int, default=24)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--tq_bits", type=int, default=4)
    parser.add_argument("--residual_len", type=int, default=128)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    rows = load_rows(args.data, args.num_requests)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    all_results: list[VLMRequestResult] = []
    summaries: list[dict[str, Any]] = []
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    concurrencies = [int(value) for value in args.concurrency.split(",") if value.strip()]
    for mode in modes:
        for concurrency in concurrencies:
            print(f"Running VLM mode={mode}, concurrency={concurrency}", flush=True)
            results, elapsed_s = run_mode_concurrency(
                model=model,
                processor=processor,
                mode=mode,
                rows=rows,
                concurrency=concurrency,
                max_new_tokens=args.max_new_tokens,
                tq_bits=args.tq_bits,
                residual_len=args.residual_len,
            )
            all_results.extend(results)
            summary = summarize(results, elapsed_s)
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    write_outputs(Path(args.output_dir), all_results, summaries)


if __name__ == "__main__":
    main()
