"""
Scan dataset/Day_1 ... Day_21, validate images, and build an Ultralytics
classification split at yolo_dataset/train and yolo_dataset/val.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import (
    DATASET_DIR,
    EXPECTED_DAYS,
    IMAGE_EXTENSIONS,
    YOLO_DATASET_DIR,
    day_folder_name,
)

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
LOW_COUNT_WARNING = 8


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width < 16 or height < 16:
                return False
        return True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False


def discover_class_folders(dataset_dir: Path) -> list[Path]:
    folders = []
    for day in EXPECTED_DAYS:
        folder = dataset_dir / day_folder_name(day)
        if folder.is_dir():
            folders.append(folder)
    extra = sorted(
        p
        for p in dataset_dir.iterdir()
        if p.is_dir()
        and p.name.lower().startswith("day_")
        and p not in folders
        and p.name.lower() != "captured"
    )
    return folders + extra


def collect_valid_images(class_folder: Path) -> tuple[list[Path], list[Path]]:
    valid = []
    skipped = []
    for path in sorted(class_folder.iterdir()):
        if not path.is_file():
            continue
        if not is_image_file(path):
            skipped.append(path)
            continue
        if is_valid_image(path):
            valid.append(path)
        else:
            skipped.append(path)
            print(f"  [SKIP] corrupted or unreadable: {path}")
    return valid, skipped


def split_images(images: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split without placing the same file in both sets."""
    shuffled = list(images)
    random.shuffle(shuffled)
    if len(shuffled) == 0:
        return [], []
    if len(shuffled) == 1:
        return shuffled, []
    train_count = int(len(shuffled) * TRAIN_RATIO)
    train_count = max(1, min(train_count, len(shuffled) - 1))
    return shuffled[:train_count], shuffled[train_count:]


def copy_unique(files: list[Path], dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in files:
        dest = dest_dir / source.name
        if dest.exists():
            dest = dest_dir / f"{source.stem}_{copied}{source.suffix}"
        shutil.copy2(source, dest)
        copied += 1
    return copied


def prepare_dataset() -> Path:
    print()
    print("==============================================")
    print(" PREPARING YOLO CLASSIFICATION DATASET")
    print("==============================================")
    print(f"Source dataset : {DATASET_DIR}")
    print(f"Output dataset : {YOLO_DATASET_DIR}")
    print(f"Random seed    : {RANDOM_SEED}")
    print()

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "Dataset folder not found.\n"
            f"Expected labeled images in:\n  {DATASET_DIR}\n"
            "or:\n  C:\\Candling\\processed\\dataset\n"
            "with subfolders Day_1 ... Day_21."
        )

    class_folders = discover_class_folders(DATASET_DIR)
    if not class_folders:
        raise FileNotFoundError(
            f"No Day_1 ... Day_21 class folders were found in:\n{DATASET_DIR}"
        )

    if YOLO_DATASET_DIR.exists():
        print("Removing previous yolo_dataset ...")
        shutil.rmtree(YOLO_DATASET_DIR)

    train_root = YOLO_DATASET_DIR / "train"
    val_root = YOLO_DATASET_DIR / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    total_train = 0
    total_val = 0
    missing_days = []
    empty_days = []
    low_days = []

    expected_names = {day_folder_name(day) for day in EXPECTED_DAYS}
    found_names = {folder.name for folder in class_folders}

    print("Images per class")
    print("----------------------------------------------")

    for day in EXPECTED_DAYS:
        name = day_folder_name(day)
        (train_root / name).mkdir(parents=True, exist_ok=True)
        (val_root / name).mkdir(parents=True, exist_ok=True)
        folder = DATASET_DIR / name
        if not folder.is_dir():
            missing_days.append(name)
            print(f"{name:8}  MISSING FOLDER")
            continue

        valid, skipped = collect_valid_images(folder)
        skip_note = f"  (skipped {len(skipped)} non-image/corrupt)" if skipped else ""
        print(f"{name:8}  {len(valid):4} valid images{skip_note}")

        if len(valid) == 0:
            empty_days.append(name)
            continue
        if len(valid) < LOW_COUNT_WARNING:
            low_days.append((name, len(valid)))

        train_images, val_images = split_images(valid)
        copied_train = copy_unique(train_images, train_root / name)
        copied_val = copy_unique(val_images, val_root / name)
        total_train += copied_train
        total_val += copied_val
        print(f"         -> {copied_train} train / {copied_val} val")

    extra = found_names - expected_names
    if extra:
        print()
        print(f"Extra class folders (included): {', '.join(sorted(extra))}")

    print("----------------------------------------------")
    print(f"TOTAL TRAIN : {total_train}")
    print(f"TOTAL VAL   : {total_val}")
    print(f"TOTAL       : {total_train + total_val}")
    print()

    if missing_days:
        print(f"[WARNING] Missing class folders: {', '.join(missing_days)}")
    if empty_days:
        print(f"[WARNING] No valid images: {', '.join(empty_days)}")
    if low_days:
        print("[WARNING] Very few images (classification will be weak):")
        for name, count in low_days:
            print(f"  - {name}: {count} image(s)")
        print(
            "  Accuracy depends on dataset size, lighting, and labeling — "
            "not on YOLO being pretrained."
        )
    if total_val == 0:
        print(
            "[WARNING] No validation images. Add at least 2 images per class "
            "so train and validation can be split without overlap."
        )
    if total_train == 0:
        raise RuntimeError(
            "Prepared dataset is empty. Add real candling images to Day_1 ... Day_21."
        )

    print()
    print(f"Ultralytics dataset ready at:\n  {YOLO_DATASET_DIR}")
    print("Next:  py train_model.py")
    return YOLO_DATASET_DIR


if __name__ == "__main__":
    prepare_dataset()
