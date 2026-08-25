from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Plate:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    text: str | None = None


class PlateDetector:
    def __init__(self, weights: Path, confidence: float, device: str = "cpu") -> None:
        if not weights.is_file():
            raise FileNotFoundError(
                f"Detector weights were not found: {weights}. Run scripts/train_detector.py first."
            )
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.device = device

    def detect(self, image: np.ndarray) -> list[Plate]:
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        height, width = image.shape[:2]
        plates: list[Plate] = []
        for coords, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = coords.astype(int).tolist()
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            if x2 > x1 and y2 > y1:
                plates.append(Plate(x1, y1, x2, y2, float(score)))
        return plates


def annotate(image: np.ndarray, plates: Iterable[Plate]) -> np.ndarray:
    result = image.copy()
    for index, plate in enumerate(plates, start=1):
        cv2.rectangle(result, (plate.x1, plate.y1), (plate.x2, plate.y2), (0, 220, 0), 3)
        label = f"plate {index}: {plate.confidence:.0%}"
        if plate.text:
            label += f"  {plate.text}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        top = max(0, plate.y1 - text_height - baseline - 8)
        cv2.rectangle(result, (plate.x1, top), (plate.x1 + text_width + 8, plate.y1), (0, 220, 0), -1)
        cv2.putText(
            result,
            label,
            (plate.x1 + 4, plate.y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return result
