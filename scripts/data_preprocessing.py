import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from config_utils import get_path, get_setting, load_project_config


VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and split image dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/project_config.json"),
        help="Project config JSON path.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None, help="Raw dataset root.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Processed dataset root.",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=None,
        help="Output JSON describing train/val split.",
    )
    parser.add_argument("--img-size", type=int, default=None, help="Image size after resizing.")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete previous processed train/val directories before writing.",
    )
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    config = load_project_config(args.config)

    args.raw_dir = args.raw_dir or get_path(config, "raw_dir", "data/raw")
    args.processed_dir = args.processed_dir or get_path(config, "processed_dir", "data/processed")
    args.splits_file = args.splits_file or get_path(config, "splits_file", "splits.json")
    args.img_size = args.img_size or int(get_setting(config, "preprocessing", "img_size", 224))
    args.val_ratio = args.val_ratio or float(get_setting(config, "preprocessing", "val_ratio", 0.1))
    args.seed = args.seed or int(get_setting(config, "preprocessing", "seed", 42))

    clean_from_config = bool(get_setting(config, "preprocessing", "clean", False))
    args.clean = args.clean or clean_from_config
    return args


def center_crop_resize(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((target_size, target_size), Image.Resampling.BILINEAR)


def collect_images(raw_dir: Path) -> Dict[str, List[Path]]:
    class_to_files: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
        )
        if files:
            class_to_files[class_dir.name] = files
    return class_to_files


def split_class_files(
    files: List[Path], val_ratio: float, rng: random.Random
) -> Tuple[List[Path], List[Path]]:
    shuffled = files[:]
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)

    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]
    return train_files, val_files


def save_processed_image(src: Path, dst: Path, img_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        rgb = image.convert("RGB")
        processed = center_crop_resize(rgb, img_size)
        processed.save(dst, format="JPEG", quality=95)


def process_split(
    split_name: str,
    class_name: str,
    files: List[Path],
    raw_dir: Path,
    processed_dir: Path,
    img_size: int,
) -> List[str]:
    records: List[str] = []
    for src in files:
        relative_src = src.relative_to(raw_dir)
        stem = relative_src.stem.replace(" ", "_")
        dst_name = f"{stem}.jpg"
        dst = processed_dir / split_name / class_name / dst_name
        save_processed_image(src, dst, img_size)
        records.append(str(dst.relative_to(processed_dir).as_posix()))
    return records


def clean_processed_dirs(processed_dir: Path) -> None:
    for split in ("train", "val"):
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()


def init_summary(args: argparse.Namespace, class_to_files: Dict[str, List[Path]]) -> dict:
    return {
        "meta": {
            "seed": args.seed,
            "img_size": args.img_size,
            "val_ratio": args.val_ratio,
            "raw_dir": str(args.raw_dir.as_posix()),
            "processed_dir": str(args.processed_dir.as_posix()),
        },
        "class_names": sorted(class_to_files.keys()),
        "splits": {"train": {}, "val": {}},
        "counts": {"train": {}, "val": {}, "total": {}},
    }


def add_class_records_to_summary(
    summary: dict,
    class_name: str,
    files: List[Path],
    train_records: List[str],
    val_records: List[str],
) -> None:
    summary["splits"]["train"][class_name] = train_records
    summary["splits"]["val"][class_name] = val_records
    summary["counts"]["train"][class_name] = len(train_records)
    summary["counts"]["val"][class_name] = len(val_records)
    summary["counts"]["total"][class_name] = len(files)


def print_summary(summary: dict, splits_file: Path) -> None:
    total_train = sum(summary["counts"]["train"].values())
    total_val = sum(summary["counts"]["val"].values())
    total_all = sum(summary["counts"]["total"].values())

    print("Preprocessing complete")
    print(f"Total images: {total_all}")
    print(f"Train images: {total_train}")
    print(f"Val images: {total_val}")
    print(f"Saved splits to: {splits_file}")


def main() -> None:
    args = resolve_settings(parse_args())

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {args.raw_dir}")

    rng = random.Random(args.seed)
    class_to_files = collect_images(args.raw_dir)

    if not class_to_files:
        raise RuntimeError(
            "No class folders with supported image files found under raw dir. "
            "Expected: data/raw/<class_name>/<image_files>."
        )

    if args.clean:
        clean_processed_dirs(args.processed_dir)

    summary = init_summary(args, class_to_files)

    for class_name, files in class_to_files.items():
        train_files, val_files = split_class_files(files, args.val_ratio, rng)

        train_records = process_split(
            "train", class_name, train_files, args.raw_dir, args.processed_dir, args.img_size
        )
        val_records = process_split(
            "val", class_name, val_files, args.raw_dir, args.processed_dir, args.img_size
        )

        add_class_records_to_summary(summary, class_name, files, train_records, val_records)

    args.splits_file.parent.mkdir(parents=True, exist_ok=True)
    with args.splits_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, args.splits_file)


if __name__ == "__main__":
    main()