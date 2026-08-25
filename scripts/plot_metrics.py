from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read_metrics(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No completed epochs in {csv_path}")
    return [
        {key.strip(): float(value) for key, value in row.items() if value not in (None, "")}
        for row in rows
    ]


def save_accuracy_chart(rows: list[dict[str, float]], output: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    series = {
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
        "mAP@50": "metrics/mAP50(B)",
        "mAP@50-95": "metrics/mAP50-95(B)",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    for label, column in series.items():
        axis.plot(epochs, [row[column] for row in rows], marker="o", linewidth=2, label=label)
    axis.set(title="Accuracy of licence-plate detector", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def save_loss_chart(rows: list[dict[str, float]], output: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    series = {
        "Train box loss": "train/box_loss",
        "Train cls loss": "train/cls_loss",
        "Train DFL loss": "train/dfl_loss",
        "Val box loss": "val/box_loss",
        "Val cls loss": "val/cls_loss",
        "Val DFL loss": "val/dfl_loss",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    for label, column in series.items():
        axis.plot(epochs, [row[column] for row in rows], marker="o", linewidth=2, label=label)
    axis.set(title="Training and validation losses", xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot detector accuracy and loss metrics.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "plate_detector")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = read_metrics(run_dir / "results.csv")
    save_accuracy_chart(rows, run_dir / "metrics_accuracy.png")
    save_loss_chart(rows, run_dir / "metrics_loss.png")
    print(f"Saved charts in {run_dir}")


if __name__ == "__main__":
    main()
