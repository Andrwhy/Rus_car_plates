from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


ALPHABET = "0123456789ABCEHKMOPTXY"
BLANK_INDEX = 0
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 192
CHAR_TO_INDEX = {character: index + 1 for index, character in enumerate(ALPHABET)}
INDEX_TO_CHAR = {index: character for character, index in CHAR_TO_INDEX.items()}

LETTER_POSITIONS = {0, 4, 5}
DIGIT_TO_LETTER = {"0": "O", "1": "T", "3": "E", "4": "A", "5": "C", "6": "B", "7": "T", "8": "B", "9": "P"}
LETTER_TO_DIGIT = {"O": "0", "A": "4", "B": "8", "C": "5", "E": "3", "H": "4", "T": "7"}


class PlateCRNN(nn.Module):
    def __init__(self, num_classes: int = len(ALPHABET) + 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.sequence = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.15)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image).mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return image


def encode_label(label: str) -> list[int]:
    return [CHAR_TO_INDEX[character] for character in label]


def ctc_decode(logits: torch.Tensor) -> list[str]:
    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded: list[str] = []
    for row in indices:
        previous = BLANK_INDEX
        characters: list[str] = []
        for index in row:
            if index != BLANK_INDEX and index != previous:
                characters.append(INDEX_TO_CHAR[index])
            previous = index
        decoded.append(normalize_russian_plate("".join(characters)))
    return decoded


def normalize_russian_plate(text: str) -> str:
    if len(text) not in {8, 9}:
        return text
    normalized: list[str] = []
    for position, character in enumerate(text):
        if position in LETTER_POSITIONS:
            normalized.append(DIGIT_TO_LETTER.get(character, character))
        else:
            normalized.append(LETTER_TO_DIGIT.get(character, character))
    candidate = "".join(normalized)
    if (
        candidate[0] in ALPHABET[10:]
        and candidate[1:4].isdigit()
        and candidate[4:6].isalpha()
        and candidate[4:6].isupper()
        and candidate[6:].isdigit()
    ):
        return candidate
    return text


def load_ocr_model(weights_path: Path, device: str) -> PlateCRNN:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = PlateCRNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
