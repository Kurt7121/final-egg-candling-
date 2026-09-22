"""
Fine-tune a pretrained YOLO classification checkpoint on Day_1 ... Day_21
chicken egg candling images. The pretrained weights do not know egg days
until this training step finishes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from config import (
    FALLBACK_PRETRAINED_CLS_MODEL,
    MODEL_PATH,
    MODELS_DIR,
    PRETRAINED_CLS_MODEL,
    RUNS_DIR,
    YOLO_DATASET_DIR,
)

# ---- configurable training settings (laptop-friendly) ----
DATASET_DIR = YOLO_DATASET_DIR
MODEL_NAME = PRETRAINED_CLS_MODEL
IMG_SIZE = 224
EPOCHS = 40
BATCH = 8
OUTPUT_DIR = RUNS_DIR
RUN_NAME = "egg_candling"
RANDOM_SEED = 42
PATIENCE = 15
WORKERS = 0


def require_prepared_dataset(dataset_dir: Path) -> list[str]:
    train_dir = dataset_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(
            "Prepared YOLO dataset not found.\n"
            f"Expected: {train_dir}\n"
            "Run this first:\n  py prepare_dataset.py"
        )

    classes = sorted(
        p.name
        for p in train_dir.iterdir()
        if p.is_dir() and any(p.iterdir())
    )
    if not classes:
        raise RuntimeError(
            f"No class folders with images found in {train_dir}.\n"
            "Run: py prepare_dataset.py"
        )

    counts = {}
    for name in classes:
        counts[name] = len([f for f in (train_dir / name).iterdir() if f.is_file()])
        if counts[name] == 0:
            print(f"[WARNING] Empty training class: {name}")

    print("Training classes:")
    for name in classes:
        print(f"  {name}: {counts[name]} train image(s)")

    missing = [f"Day_{d}" for d in range(1, 22) if f"Day_{d}" not in classes]
    if missing:
        print(
            "[WARNING] Missing classes in the prepared training set:\n  "
            + ", ".join(missing)
        )
        print("The model can only learn classes that exist in the dataset.")

    return classes


def load_base_model():
    from ultralytics import YOLO

    try:
        print(f"Loading pretrained classification checkpoint: {MODEL_NAME}")
        print(
            "This checkpoint is a starting point only. It has not been trained "
            "on chicken egg candling images yet."
        )
        return YOLO(MODEL_NAME)
    except Exception as error:
        print(f"Could not load {MODEL_NAME}: {error}")
        print(f"Trying fallback checkpoint: {FALLBACK_PRETRAINED_CLS_MODEL}")
        return YOLO(FALLBACK_PRETRAINED_CLS_MODEL)


def pick_device() -> str | int:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def save_best_weights(run_dir: Path) -> Path:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    source = best if best.exists() else last
    if not source.exists():
        raise FileNotFoundError(f"Training did not produce weights in {run_dir / 'weights'}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, MODEL_PATH)
    print(f"Copied trained weights:\n  {source}\n  -> {MODEL_PATH}")
    return MODEL_PATH


def train() -> None:
    from ultralytics import YOLO

    print()
    print("======================================================")
    print(" CHICKEN EGG CANDLING — YOLO CLASSIFICATION TRAINING")
    print("======================================================")
    print(f"DATASET_DIR : {DATASET_DIR}")
    print(f"MODEL_NAME  : {MODEL_NAME}")
    print(f"IMG_SIZE    : {IMG_SIZE}")
    print(f"EPOCHS      : {EPOCHS}")
    print(f"BATCH       : {BATCH}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"FINAL MODEL : {MODEL_PATH}")
    print("======================================================")
    print()

    classes = require_prepared_dataset(DATASET_DIR)
    device = pick_device()
    print(f"Device: {device}")
    print()

    model = load_base_model()
    results = model.train(
        data=str(DATASET_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        seed=RANDOM_SEED,
        patience=PATIENCE,
        workers=WORKERS,
        device=device,
        pretrained=True,
    )

    run_dir = Path(results.save_dir) if getattr(results, "save_dir", None) else OUTPUT_DIR / RUN_NAME
    print()
    print("Training finished. Save directory:")
    print(f"  {run_dir}")

    print()
    print("Validation metrics from training:")
    try:
        metrics = getattr(results, "top1", None)
        top1 = getattr(results, "top1", None)
        top5 = getattr(results, "top5", None)
        if hasattr(results, "results_dict"):
            print(results.results_dict)
        if top1 is not None:
            print(f"  top1: {top1}")
        if top5 is not None:
            print(f"  top5: {top5}")
        if top1 is None and metrics is None:
            print("  (see results CSV under the run folder)")
    except Exception as error:
        print(f"  Could not print inline metrics: {error}")
        print("  Check the Ultralytics results folder for actual numbers.")

    print()
    print("Running an extra validation pass on the prepared val split...")
    try:
        trained = YOLO(str(run_dir / "weights" / "best.pt"))
        val_metrics = trained.val(
            data=str(DATASET_DIR),
            imgsz=IMG_SIZE,
            batch=BATCH,
            workers=WORKERS,
            device=device,
        )
        print("Validation result object:")
        print(val_metrics)
        top1 = getattr(val_metrics, "top1", None)
        top5 = getattr(val_metrics, "top5", None)
        if top1 is not None:
            print(f"Validation top-1 accuracy: {float(top1):.4f}")
        if top5 is not None:
            print(f"Validation top-5 accuracy: {float(top5):.4f}")
        print("Do not assume this is high — it depends on this dataset.")
        print("Final class names:")
        for index, name in trained.names.items():
            print(f"  {index}: {name}")
    except Exception as error:
        print(f"Validation pass failed: {error}")

    save_best_weights(run_dir)
    print()
    print("Fine-tuned chicken egg candling model saved to:")
    print(f"  {MODEL_PATH}")
    print("This file is what the web app must load — not the pretrained base.")
    print("Classes expected in training:", ", ".join(classes))


if __name__ == "__main__":
    try:
        train()
    except Exception as error:
        print()
        print("======================================================")
        print(" TRAINING ERROR")
        print("======================================================")
        print(error)
        raise SystemExit(1)
