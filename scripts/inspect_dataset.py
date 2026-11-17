"""Inspect an image-folder dataset without training or changing it."""
import argparse
from collections import Counter
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect(root: Path) -> dict[str, object]:
    counts = Counter(
        p.parent.name for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    return {"root": str(root), "images": sum(counts.values()), "classes": dict(sorted(counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect class folders under DATASET.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.dataset}")
    result = inspect(args.dataset)
    print(f"Dataset: {result['root']}\nImages: {result['images']}")
    for label, count in result["classes"].items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
