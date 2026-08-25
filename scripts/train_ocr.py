from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.plate_ocr_model import ALPHABET, IMAGE_HEIGHT, IMAGE_WIDTH, PlateCRNN, ctc_decode, encode_label, preprocess_plate


DEFAULT_DATA_DIR = ROOT / "data" / "car_plate_ocr"


class OCRDataset(Dataset):
    def __init__(self, directory: Path, augment: bool, cache_images: bool) -> None:
        self.files = sorted(directory.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {directory}")
        self.labels = [file.stem for file in self.files]
        self.augment = augment
        self.cache: np.ndarray | None = None
        if cache_images:
            self.cache = np.empty((len(self.files), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
            for index, file in enumerate(self.files):
                self.cache[index] = self._read(file)

    @staticmethod
    def _read(file: Path) -> np.ndarray:
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable OCR image: {file}")
        return preprocess_plate(image)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image = self.cache[index].copy() if self.cache is not None else self._read(self.files[index])
        if self.augment:
            if random.random() < 0.7:
                image = cv2.convertScaleAbs(image, alpha=random.uniform(0.75, 1.25), beta=random.randint(-28, 28))
            if random.random() < 0.18:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            if random.random() < 0.2:
                noise = np.random.normal(0, random.uniform(2, 10), image.shape)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(image).unsqueeze(0).float().div(255)
        return tensor, self.labels[index]


def collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    images, labels = zip(*batch)
    encoded = [torch.tensor(encode_label(label), dtype=torch.long) for label in labels]
    return (
        torch.stack(images),
        torch.cat(encoded),
        torch.tensor([len(label) for label in labels], dtype=torch.long),
        list(labels),
    )


def validate(model: PlateCRNN, loader: DataLoader, criterion: nn.CTCLoss, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    exact = 0
    characters = 0
    correct_characters = 0
    batches = 0
    with torch.inference_mode():
        for images, targets, target_lengths, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            predictions = ctc_decode(logits)
            total_loss += loss.item()
            batches += 1
            exact += sum(prediction == label for prediction, label in zip(predictions, labels))
            for prediction, label in zip(predictions, labels):
                correct_characters += sum(a == b for a, b in zip(prediction, label))
                characters += len(label)
    return total_loss / batches, exact / len(loader.dataset), correct_characters / characters


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCR specialised for Russian vehicle plates.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cache-images", action="store_true")
    parser.add_argument("--run-name", default="plate_ocr")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    device = torch.device(args.device)
    train_data = OCRDataset(data_dir / "train" / "img", augment=True, cache_images=not args.no_cache_images)
    val_data = OCRDataset(data_dir / "val" / "img", augment=False, cache_images=not args.no_cache_images)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", collate_fn=collate)

    model = PlateCRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "ocr_metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "exact_accuracy", "character_accuracy", "learning_rate"]
    best_accuracy = -1.0
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for images, targets, target_lengths, _ in train_loader:
                images = images.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(images)
                    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                    input_lengths = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            val_loss, exact_accuracy, character_accuracy = validate(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / len(train_loader),
                "val_loss": val_loss,
                "exact_accuracy": exact_accuracy,
                "character_accuracy": character_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            file.flush()
            print(
                f"epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.4f} "
                f"val_loss={val_loss:.4f} exact={exact_accuracy:.2%} char={character_accuracy:.2%}"
            )
            if exact_accuracy > best_accuracy:
                best_accuracy = exact_accuracy
                output = ROOT / "models" / "plate_ocr_crnn.pt"
                output.parent.mkdir(exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "alphabet": ALPHABET}, output)
            scheduler.step()
    print(f"Saved best OCR weights to {ROOT / 'models' / 'plate_ocr_crnn.pt'}")


if __name__ == "__main__":
    main()
