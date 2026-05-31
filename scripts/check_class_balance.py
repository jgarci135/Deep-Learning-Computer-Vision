import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from config_utils import get_path, load_project_config


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze class balance and suggest class weights.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--splits-file", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    config = load_project_config(args.config)
    args.raw_dir = args.raw_dir or get_path(config, "raw_dir", "data/raw")
    args.processed_dir = args.processed_dir or get_path(config, "processed_dir", "data/processed")
    args.splits_file = args.splits_file or get_path(config, "splits_file", "splits.json")
    args.figures_dir = args.figures_dir or get_path(config, "figures_dir", "Figures")
    args.models_dir = args.models_dir or get_path(config, "models_dir", "models")
    return args


def list_image_count(class_dir: Path) -> int:
    return sum(1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)


def collect_split_counts(split_dir: Path) -> Dict[str, int]:
    if not split_dir.exists():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = list_image_count(class_dir)
    return counts


def collect_raw_counts(raw_dir: Path) -> Dict[str, int]:
    if not raw_dir.exists():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = list_image_count(class_dir)
    return counts


def suggest_class_weights(train_counts: Dict[str, int]) -> Dict[str, float]:
    if not train_counts:
        return {}
    max_count = max(train_counts.values())
    return {cls: round(max_count / count, 4) if count > 0 else 0.0 for cls, count in train_counts.items()}


def compute_imbalance_ratio(counts: Dict[str, int]) -> float:
    positive_counts = [c for c in counts.values() if c > 0]
    if not positive_counts:
        return 0.0
    return max(positive_counts) / min(positive_counts)


def plot_counts(train_counts: Dict[str, int], val_counts: Dict[str, int], out_path: Path) -> None:
    labels: List[str] = sorted(set(train_counts.keys()).union(val_counts.keys()))
    train = [train_counts.get(k, 0) for k in labels]
    val = [val_counts.get(k, 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.38

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, train, width=width, label="train")
    plt.bar(x + width / 2, val, width=width, label="val")
    plt.xticks(x, labels)
    plt.ylabel("Image count")
    plt.title("Class Distribution by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = resolve_paths(parse_args())

    train_counts = collect_split_counts(args.processed_dir / "train")
    val_counts = collect_split_counts(args.processed_dir / "val")
    raw_counts = collect_raw_counts(args.raw_dir)

    if not train_counts and raw_counts:
        train_counts = raw_counts

    if not train_counts:
        raise RuntimeError(
            "No class counts found. Expected images under data/processed/train or data/raw."
        )

    class_weights = suggest_class_weights(train_counts)
    imbalance_ratio = compute_imbalance_ratio(train_counts)

    plot_path = args.figures_dir / "class_distribution.png"
    plot_counts(train_counts, val_counts, plot_path)

    report = {
        "counts": {
            "raw": raw_counts,
            "train": train_counts,
            "val": val_counts,
        },
        "imbalance_ratio_train_max_to_min": round(imbalance_ratio, 4),
        "suggested_class_weights": class_weights,
    }

    args.models_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.models_dir / "class_balance_report.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved class balance plot: {plot_path}")
    print(f"Saved class balance report: {output_path}")
    print(f"Train imbalance ratio (max/min): {imbalance_ratio:.4f}")


if __name__ == "__main__":
    main()
