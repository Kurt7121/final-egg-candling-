"""Shared paths and settings for the egg candling system."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Labeled images currently live in processed/dataset (Day_1 ... Day_21).
# If a top-level dataset/ folder with those classes is added later, it is preferred.
_CANDIDATE_DATASETS = [
    BASE_DIR / "dataset",
    BASE_DIR / "processed" / "dataset",
]


def resolve_dataset_dir() -> Path:
    for candidate in _CANDIDATE_DATASETS:
        if (candidate / "Day_1").is_dir() or (candidate / "Day_21").is_dir():
            return candidate
    return _CANDIDATE_DATASETS[0]


DATASET_DIR = resolve_dataset_dir()
YOLO_DATASET_DIR = BASE_DIR / "yolo_dataset"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "egg_candling_best.pt"
CAPTURED_DIR = DATASET_DIR / "captured"
DB_PATH = BASE_DIR / "egg_candling.db"
RUNS_DIR = BASE_DIR / "runs"

# Do not treat a generic pretrained YOLO checkpoint as the egg model.
PRETRAINED_CLS_MODEL = "yolo11n-cls.pt"
FALLBACK_PRETRAINED_CLS_MODEL = "yolov8n-cls.pt"

MIN_CONFIDENCE = 0.60
MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64
DARK_MEAN_THRESHOLD = 18.0
LOW_PIXEL_COUNT_THRESHOLD = 100

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DAY_CLASS_PREFIX = "Day_"
EXPECTED_DAYS = list(range(1, 22))


def day_folder_name(day: int) -> str:
    return f"{DAY_CLASS_PREFIX}{day}"


def display_day_name(class_name: str) -> str:
    """Normalize Day_14 / Day 14 / day14 into 'Day 14'."""
    if not class_name:
        return ""
    text = str(class_name).strip().replace("-", "_")
    lowered = text.lower().replace(" ", "_")
    if lowered.startswith("day_"):
        suffix = lowered[4:]
        if suffix.isdigit():
            return f"Day {int(suffix)}"
    if lowered.startswith("day") and lowered[3:].strip("_ ").isdigit():
        return f"Day {int(lowered[3:].strip('_ '))}"
    return text.replace("_", " ")


def class_folder_from_display(day_label: str) -> str:
    """Convert 'Day 14' or 'Day_14' into folder name Day_14."""
    pretty = display_day_name(day_label)
    parts = pretty.split()
    if len(parts) == 2 and parts[1].isdigit():
        return f"Day_{int(parts[1])}"
    return day_label.replace(" ", "_")
