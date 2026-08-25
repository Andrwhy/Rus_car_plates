from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from app.plate_ocr_model import ctc_decode, load_ocr_model, preprocess_plate


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_MODEL_PATH = ROOT / "models" / "plate_ocr_crnn.pt"

# Used only when the specialised model has not been trained yet.
ALLOWED_CHARACTERS = "ABEKMHOPCTYX0123456789"
_CYRILLIC_TO_LATIN = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)


def _device() -> str:
    """Match the bot's DEVICE setting while accepting Ultralytics-style `0`."""
    configured = os.getenv("DEVICE", "cpu").strip()
    return f"cuda:{configured}" if configured.isdigit() else configured


@lru_cache(maxsize=1)
def _custom_model():
    if not CUSTOM_MODEL_PATH.is_file():
        return None
    return load_ocr_model(CUSTOM_MODEL_PATH, _device())


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _read_custom(crop: np.ndarray) -> str:
    image = preprocess_plate(crop)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().div(255)
    model = _custom_model()
    if model is None:  # Guard for direct use during deployment.
        return ""
    device = next(model.parameters()).device
    with torch.inference_mode():
        return ctc_decode(model(tensor.to(device)))[0]


def _read_easyocr(crop: np.ndarray) -> str:
    readings = _easyocr_reader().readtext(crop, detail=1, allowlist=ALLOWED_CHARACTERS)
    if not readings:
        return ""
    text = "".join(item[1] for item in readings).upper().translate(_CYRILLIC_TO_LATIN)
    return re.sub(f"[^{ALLOWED_CHARACTERS}]", "", text)


def read_plate(crop: np.ndarray) -> str | None:
    """Read a localised Russian plate with the specialised CRNN when available."""
    text = _read_custom(crop) if CUSTOM_MODEL_PATH.is_file() else _read_easyocr(crop)
    return text or None
