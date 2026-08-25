from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    token: str
    model_path: Path
    confidence: float
    device: str
    enable_ocr: bool


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing. Create .env and add the token from @BotFather."
        )

    model_path = Path(os.getenv("MODEL_PATH", "models/license_plate_detector.pt"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    try:
        confidence = float(os.getenv("CONFIDENCE", "0.35"))
    except ValueError as exc:
        raise RuntimeError("CONFIDENCE must be a number between 0 and 1.") from exc
    if not 0 < confidence < 1:
        raise RuntimeError("CONFIDENCE must be a number between 0 and 1.")

    return Settings(
        token=token,
        model_path=model_path,
        confidence=confidence,
        device=os.getenv("DEVICE", "cpu"),
        enable_ocr=os.getenv("ENABLE_OCR", "false").strip().lower() in {"1", "true", "yes"},
    )
