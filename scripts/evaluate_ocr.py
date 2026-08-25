from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.plate_ocr_model import ctc_decode, load_ocr_model, preprocess_plate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the specialised plate OCR on a held-out split.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "car_plate_ocr" / "test" / "img")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "plate_ocr_crnn.pt")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    files = sorted(args.data_dir.resolve().glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No OCR images found in {args.data_dir}")
    device = args.device
    model = load_ocr_model(args.weights.resolve(), device)
    exact = correct_characters = total_characters = 0

    with torch.inference_mode():
        for start in range(0, len(files), args.batch):
            batch = files[start : start + args.batch]
            images = np.stack(
                [preprocess_plate(cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)) for file in batch]
            )
            logits = model(torch.from_numpy(images).unsqueeze(1).float().div(255).to(device))
            predictions = ctc_decode(logits)
            labels = [file.stem for file in batch]
            exact += sum(prediction == label for prediction, label in zip(predictions, labels))
            for prediction, label in zip(predictions, labels):
                correct_characters += sum(a == b for a, b in zip(prediction, label))
                total_characters += len(label)

    print(
        f"test_images={len(files)} exact_accuracy={exact / len(files):.2%} "
        f"character_accuracy={correct_characters / total_characters:.2%}"
    )


if __name__ == "__main__":
    main()
