import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = [
    ("red", "A", (220, 40, 40)),
    ("green", "B", (40, 180, 80)),
    ("blue", "C", (40, 80, 220)),
    ("yellow", "D", (230, 200, 40)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/synthetic_vlm")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "requests.jsonl"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx in range(args.limit):
            color_name, answer, rgb = COLORS[idx % len(COLORS)]
            image_path = image_dir / f"{idx:06d}.jpg"

            image = Image.new("RGB", (512, 512), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((96, 96, 416, 416), fill=rgb)
            draw.text((180, 235), color_name.upper(), fill="black")
            image.save(image_path)

            prompt = (
                "Look at the image. What is the main color of the square?\n"
                "Options: A: red, B: green, C: blue, D: yellow.\n"
                "Only output a single letter."
            )
            row = {
                "id": idx,
                "image_path": str(image_path.resolve()),
                "prompt": prompt,
                "answer": answer,
                "category": "synthetic_smoke",
                "l2_category": "color_recognition",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved: {out_jsonl}")


if __name__ == "__main__":
    main()
