import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def collect_gpu_peaks(log_dir: Path) -> dict[str, dict[str, float]]:
    peaks: dict[str, dict[str, float]] = {}
    for path in log_dir.glob("*_gpu.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        key = path.name.replace("_gpu.csv", "")
        peaks[key] = {
            "gpu_memory_peak_mib": float(df["memory.used"].max()),
            "gpu_memory_mean_mib": float(df["memory.used"].mean()),
            "gpu_util_mean_pct": float(df["utilization.gpu"].mean()),
        }
    return peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--out", type=str, default="results/summary_all.csv")
    args = parser.parse_args()

    rows = []
    gpu_peaks = collect_gpu_peaks(Path(args.logs_dir))
    for path in glob.glob(str(Path(args.results_dir) / "*.summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["file"] = path

        stem = Path(path).name.replace(".summary.json", "")
        job_model = "_".join(stem.split("_")[:2]) if "_" in stem else ""
        for gpu_key, gpu_row in gpu_peaks.items():
            if stem.startswith(gpu_key) or job_model and gpu_key.startswith(job_model):
                row.update(gpu_row)
                break
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No summary found.")
        return

    preferred_cols = [
        "model",
        "concurrency",
        "num_prompts",
        "max_tokens",
        "requests_per_s",
        "output_units_per_s",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "ttft_mean_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_mean_s",
        "tpot_p95_s",
        "accuracy",
        "parse_success_rate",
        "error_rate",
        "gpu_memory_peak_mib",
        "gpu_memory_mean_mib",
        "gpu_util_mean_pct",
        "file",
    ]
    cols = [col for col in preferred_cols if col in df.columns]
    df = df[cols].sort_values(["model", "concurrency"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
