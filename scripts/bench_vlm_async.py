import argparse
import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from tqdm import tqdm


def image_to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def parse_choice(text: str) -> str | None:
    if not text:
        return None
    normalized = text.strip().upper()
    match = re.search(r"\b([ABCD])\b", normalized)
    if match:
        return match.group(1)
    if normalized and normalized[0] in "ABCD":
        return normalized[0]
    return None


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(v for v in values if pd.notna(v))
    if not clean:
        return None
    idx = min(len(clean) - 1, round((q / 100) * (len(clean) - 1)))
    return float(clean[idx])


def usage_total_tokens(usage: dict[str, Any] | None, key: str) -> int | None:
    if not usage:
        return None
    value = usage.get(key)
    return int(value) if value is not None else None


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    item: dict[str, Any],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(item["image_path"])}},
                    {"type": "text", "text": item["prompt"]},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time = None
    text_parts: list[str] = []
    completion_chunks = 0
    usage = None
    error = None

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                error = f"HTTP_{response.status}: {await response.text()}"
            else:
                async for raw in response.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("usage"):
                        usage = obj["usage"]

                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        completion_chunks += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        text_parts.append(token)
    except Exception as exc:
        error = repr(exc)

    end = time.perf_counter()
    output_text = "".join(text_parts).strip()
    pred = parse_choice(output_text)
    gold = str(item.get("answer", "")).strip().upper()[:1]
    completion_tokens = usage_total_tokens(usage, "completion_tokens")
    prompt_tokens = usage_total_tokens(usage, "prompt_tokens")
    measured_output_units = completion_tokens or completion_chunks or None
    ttft = (first_token_time - start) if first_token_time else None
    latency = end - start
    decode_time = (latency - ttft) if ttft is not None else None
    tpot = (decode_time / max(measured_output_units - 1, 1)) if decode_time and measured_output_units else None

    return {
        "id": item.get("id"),
        "latency_s": latency,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "output_text": output_text,
        "pred": pred,
        "gold": gold,
        "correct": (pred == gold) if gold else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_chunks": completion_chunks,
        "error": error,
    }


async def run(args: argparse.Namespace) -> None:
    with open(args.data, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = rows[: args.num_prompts]

    sem = asyncio.Semaphore(args.concurrency)
    url = args.base_url.rstrip("/") + "/chat/completions"

    async with aiohttp.ClientSession() as session:
        async def bounded(item: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await one_request(session, url, args.model, item, args.max_tokens, args.timeout)

        start = time.perf_counter()
        tasks = [asyncio.create_task(bounded(item)) for item in rows]
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"c={args.concurrency}"):
            results.append(await task)
        end = time.perf_counter()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)
    total_time = end - start
    success_df = df[df["error"].isna()].copy()
    completion_tokens = int(success_df["completion_tokens"].dropna().sum()) if "completion_tokens" in success_df else 0
    completion_units = completion_tokens or int(success_df["completion_chunks"].dropna().sum())

    summary = {
        "model": args.model,
        "data": args.data,
        "num_prompts": len(rows),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "total_time_s": total_time,
        "requests_per_s": len(rows) / total_time if total_time > 0 else None,
        "output_units_per_s": completion_units / total_time if total_time > 0 else None,
        "success": int(df["error"].isna().sum()),
        "errors": int(df["error"].notna().sum()),
        "error_rate": float(df["error"].notna().mean()),
        "accuracy": float(success_df["correct"].mean()) if len(success_df) else None,
        "parse_success_rate": float(success_df["pred"].notna().mean()) if len(success_df) else None,
        "latency_mean_s": float(success_df["latency_s"].mean()) if len(success_df) else None,
        "latency_p50_s": percentile(success_df["latency_s"].tolist(), 50) if len(success_df) else None,
        "latency_p95_s": percentile(success_df["latency_s"].tolist(), 95) if len(success_df) else None,
        "latency_p99_s": percentile(success_df["latency_s"].tolist(), 99) if len(success_df) else None,
        "ttft_mean_s": float(success_df["ttft_s"].dropna().mean()) if len(success_df) else None,
        "ttft_p50_s": percentile(success_df["ttft_s"].dropna().tolist(), 50) if len(success_df) else None,
        "ttft_p95_s": percentile(success_df["ttft_s"].dropna().tolist(), 95) if len(success_df) else None,
        "tpot_mean_s": float(success_df["tpot_s"].dropna().mean()) if len(success_df) else None,
        "tpot_p95_s": percentile(success_df["tpot_s"].dropna().tolist(), 95) if len(success_df) else None,
        "token_source": "usage.completion_tokens" if completion_tokens else "stream_chunks",
    }

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=str, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
