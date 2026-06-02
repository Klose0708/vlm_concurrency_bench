import argparse
import io
import json
from pathlib import Path

from datasets import load_dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm


def save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.convert("RGB").save(path)
    elif isinstance(image, bytes):
        Image.open(io.BytesIO(image)).convert("RGB").save(path)
    else:
        Image.open(image).convert("RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/mmstar")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument(
        "--local_parquet",
        type=str,
        default="",
        help="Optional local MMStar parquet file, e.g. /data/private/$USER/workspace/datasets/MMStar/mmstar.parquet",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    out_jsonl = out_dir / "mmstar_requests.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.local_parquet:
        dataframe = pd.read_parquet(args.local_parquet)
        if args.limit > 0:
            dataframe = dataframe.iloc[: args.limit]
        dataset = dataframe.to_dict("records")
    else:
        dataset = load_dataset("Lin-Chen/MMStar", split=args.split)
        if args.limit > 0:
            dataset = dataset.select(range(min(args.limit, len(dataset))))

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(dataset, desc="Preparing MMStar")):
            image_path = image_dir / f"{idx:06d}.jpg"
            save_image(item["image"], image_path)

            question = item.get("question", "")
            prompt = (
                "Look at the image and answer the multiple-choice question.\n"
                "You must choose one option from A, B, C, or D.\n"
                "Only output a single letter.\n\n"
                f"{question}"
            )

            row = {
                "id": idx,
                "image_path": str(image_path.resolve()),
                "prompt": prompt,
                "answer": str(item.get("answer", "")).strip(),
                "category": item.get("category", ""),
                "l2_category": item.get("l2_category", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved: {out_jsonl}")


if __name__ == "__main__":
    main()
