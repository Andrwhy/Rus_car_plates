from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the local OCR dataset layout before training an OCR model.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "car_plate_ocr")
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"OCR dataset is missing: {directory}")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [path for path in directory.rglob("*") if path.suffix.lower() in extensions]
    print(f"Images: {len(images)}")
    print("Images by parent folder:")
    for folder, count in Counter(path.parent.name for path in images).most_common():
        print(f"  {folder}: {count}")
    metadata = [path for path in directory.rglob("*") if path.suffix.lower() in {".csv", ".json", ".txt"}]
    print("Possible annotation files:")
    for path in metadata[:20]:
        print(f"  {path.relative_to(directory)}")


if __name__ == "__main__":
    main()
