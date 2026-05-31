import argparse
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from config_utils import get_path, get_setting, load_project_config


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA figures for local dataset.")
    parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--hist-bins", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    config = load_project_config(args.config)
    args.processed_dir = args.processed_dir or get_path(config, "processed_dir", "data/processed")
    args.figures_dir = args.figures_dir or get_path(config, "figures_dir", "Figures")
    args.max_images_per_class = args.max_images_per_class or int(
        get_setting(config, "eda", "max_images_per_class", 4)
    )
    args.hist_bins = args.hist_bins or int(get_setting(config, "eda", "hist_bins", 64))
    args.seed = args.seed or int(get_setting(config, "eda", "seed", 42))
    return args


def list_images(class_dir: Path) -> List[Path]:
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES)


def collect_counts(split_dir: Path) -> Dict[str, int]:
    if not split_dir.exists():
        return {}
    counts: Dict[str, int] = {}
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = len(list_images(class_dir))
    return counts


def plot_class_distribution(train_counts: Dict[str, int], val_counts: Dict[str, int], out_path: Path) -> None:
    labels = sorted(set(train_counts.keys()).union(val_counts.keys()))
    train_vals = [train_counts.get(lbl, 0) for lbl in labels]
    val_vals = [val_counts.get(lbl, 0) for lbl in labels]

    x = np.arange(len(labels))
    width = 0.4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, train_vals, width, label="train", color="#4472c4")
    plt.bar(x + width / 2, val_vals, width, label="val", color="#70ad47")
    plt.xticks(x, labels)
    plt.ylabel("Image count")
    plt.title("Train vs Validation Class Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sample_grid(train_dir: Path, max_images_per_class: int, seed: int, out_path: Path) -> None:
    class_dirs = [p for p in sorted(train_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        return

    random.seed(seed)
    cols = max_images_per_class
    rows = len(class_dirs)

    plt.figure(figsize=(3 * cols, 3 * rows))
    plot_idx = 1

    for class_dir in class_dirs:
        files = list_images(class_dir)
        if not files:
            for _ in range(cols):
                plt.subplot(rows, cols, plot_idx)
                plt.axis("off")
                plot_idx += 1
            continue

        sampled = files[:]
        random.shuffle(sampled)
        sampled = sampled[:cols]

        if len(sampled) < cols:
            sampled = sampled + sampled[:1] * (cols - len(sampled))

        for img_path in sampled:
            with Image.open(img_path) as img:
                arr = np.array(img.convert("RGB"))
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(arr)
            plt.title(class_dir.name)
            plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_histograms(train_dir: Path, hist_bins: int, out_path: Path) -> None:
    class_dirs = [p for p in sorted(train_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        return

    plt.figure(figsize=(10, 5))
    for class_dir in class_dirs:
        files = list_images(class_dir)
        if not files:
            continue
        with Image.open(files[0]) as img:
            gray = np.array(img.convert("L"))
        plt.hist(gray.ravel(), bins=hist_bins, alpha=0.35, label=class_dir.name)

    plt.title("Grayscale Pixel Intensity Histogram by Class")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = resolve_settings(parse_args())

    train_dir = args.processed_dir / "train"
    val_dir = args.processed_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            "Training data not found under processed directory. Run preprocessing first."
        )

    train_counts = collect_counts(train_dir)
    val_counts = collect_counts(val_dir)

    plot_class_distribution(train_counts, val_counts, args.figures_dir / "eda_class_distribution.png")
    plot_sample_grid(
        train_dir,
        args.max_images_per_class,
        args.seed,
        args.figures_dir / "eda_sample_grid.png",
    )
    plot_histograms(train_dir, args.hist_bins, args.figures_dir / "eda_histograms.png")

    print(f"Saved: {args.figures_dir / 'eda_class_distribution.png'}")
    print(f"Saved: {args.figures_dir / 'eda_sample_grid.png'}")
    print(f"Saved: {args.figures_dir / 'eda_histograms.png'}")


if __name__ == "__main__":
    main()
