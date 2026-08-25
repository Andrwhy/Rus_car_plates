from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "car_plate_detecting"
SPLITS = ("train", "val", "test")


def extract_archives(dataset_dir: Path) -> None:
    """Extract each official split once, preserving its images/labels layout."""
    for split in SPLITS:
        split_dir = dataset_dir / split
        archive = dataset_dir / f"{split}.zip"
        if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
            continue
        if not archive.is_file():
            raise FileNotFoundError(f"Missing dataset archive: {archive}")
        with zipfile.ZipFile(archive) as content:
            for member in content.infolist():
                target = (dataset_dir / member.filename).resolve()
                if not target.is_relative_to(dataset_dir.resolve()):
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            content.extractall(dataset_dir)


def write_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    project_path = dataset_dir.resolve().as_posix()
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {project_path}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: license_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def prepare(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {dataset_dir}")
    extract_archives(dataset_dir)
    return write_data_yaml(dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack the YOLO plate-detection data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    print(f"Prepared dataset config: {prepare(args.data_dir)}")


if __name__ == "__main__":
    main()
