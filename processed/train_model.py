import os
import shutil
import random
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

# Automatically use the folder where train_model.py is located
BASE_DIR = Path(__file__).resolve().parent

# Your original dataset
SOURCE_DATASET = BASE_DIR / "dataset"

# Temporary YOLO classification dataset
YOLO_DATASET = BASE_DIR / "yolo_dataset"

# Where YOLO training results will be saved
RUNS_DIR = BASE_DIR / "runs"

# Final trained model
MODEL_DIR = BASE_DIR / "model"
FINAL_MODEL = MODEL_DIR / "egg_candling_yolo.pt"

# Number of stages
STAGES = [f"Day_{i}" for i in range(1, 22)]

# Train/validation split
TRAIN_RATIO = 0.80

# Random seed
RANDOM_SEED = 42

# Training settings
EPOCHS = 50
IMAGE_SIZE = 224
BATCH_SIZE = 16