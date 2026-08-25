from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
from prepare_detection_dataset import prepare


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "car_plate_detecting"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO licence-plate detector.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="yolo11n.pt", help="Base Ultralytics model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 0,1, ...")
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name", default="plate_detector")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is the reliable setting for this Windows environment.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    data_yaml = prepare(data_dir)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.run_name,
        exist_ok=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best weights were not found: {best}")
    destination = ROOT / "models" / "license_plate_detector.pt"
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(best, destination)
    print(f"Saved bot weights to {destination}")


if __name__ == "__main__":
    main()
