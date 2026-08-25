from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot custom plate OCR metrics.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "plate_ocr")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "ocr_metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    if not rows:
        raise ValueError("No OCR metrics are available.")
    epochs = [row["epoch"] for row in rows]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, [row["exact_accuracy"] for row in rows], marker="o", linewidth=2, label="Exact plate")
    axis.plot(epochs, [row["character_accuracy"] for row in rows], marker="o", linewidth=2, label="Character")
    axis.set(title="Russian plate OCR accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "ocr_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, [row["train_loss"] for row in rows], marker="o", linewidth=2, label="Train CTC loss")
    axis.plot(epochs, [row["val_loss"] for row in rows], marker="o", linewidth=2, label="Validation CTC loss")
    axis.set(title="Russian plate OCR loss", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "ocr_loss.png", dpi=180)
    plt.close(figure)
    print(f"Saved OCR charts in {run_dir}")


if __name__ == "__main__":
    main()
