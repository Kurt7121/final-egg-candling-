import os
import shutil
import random
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(r"C:\Candling")

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_image_file(file):
    return file.suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]


def prepare_dataset():
    print("\n==============================================")
    print(" PREPARING DATASET")
    print("==============================================\n")

    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{SOURCE_DATASET}"
        )

    # Remove old prepared dataset
    if YOLO_DATASET.exists():
        print("Removing old yolo_dataset...")
        shutil.rmtree(YOLO_DATASET)

    # Create train and val folders
    train_dir = YOLO_DATASET / "train"
    val_dir = YOLO_DATASET / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)

    total_train = 0
    total_val = 0

    print("Reading Day 1 to Day 21...\n")

    for stage in STAGES:

        source_folder = SOURCE_DATASET / stage

        if not source_folder.exists():
            print(f"[WARNING] Missing folder: {source_folder}")
            continue

        # Get images
        images = [
            file
            for file in source_folder.iterdir()
            if file.is_file() and is_image_file(file)
        ]

        if len(images) == 0:
            print(f"[WARNING] No images found in {stage}")
            continue

        # Shuffle images
        random.shuffle(images)

        # Calculate training images
        train_count = int(len(images) * TRAIN_RATIO)

        # Make sure validation has at least one image
        if len(images) > 1 and train_count >= len(images):
            train_count = len(images) - 1

        train_images = images[:train_count]
        val_images = images[train_count:]

        # Create class folders
        train_stage_folder = train_dir / stage
        val_stage_folder = val_dir / stage

        train_stage_folder.mkdir(parents=True, exist_ok=True)
        val_stage_folder.mkdir(parents=True, exist_ok=True)

        # Copy training images
        for image in train_images:
            destination = train_stage_folder / image.name
            shutil.copy2(image, destination)

        # Copy validation images
        for image in val_images:
            destination = val_stage_folder / image.name
            shutil.copy2(image, destination)

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"{stage}: "
            f"{len(images)} total | "
            f"{len(train_images)} train | "
            f"{len(val_images)} validation"
        )

    print("\n----------------------------------------------")
    print(f"TOTAL TRAINING IMAGES   : {total_train}")
    print(f"TOTAL VALIDATION IMAGES : {total_val}")
    print(f"TOTAL IMAGES            : {total_train + total_val}")
    print("----------------------------------------------\n")

    if total_train == 0:
        raise RuntimeError(
            "No training images were found. "
            "Check dataset/Day_1 ... Day_21."
        )

    if total_val == 0:
        raise RuntimeError(
            "No validation images were created. "
            "You need more than one image per class."
        )

    return train_dir, val_dir


def train_model(train_dir, val_dir):

    print("\n==============================================")
    print(" STARTING YOLO TRAINING")
    print("==============================================\n")

    print("Loading YOLO classification model...")

    # YOLO classification model
    model = YOLO("yolo26n-cls.pt")

    print("\nStarting training...")
    print("This may take a while.\n")

    results = model.train(
        data=str(YOLO_DATASET),

        # Training
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # Output
        project=str(RUNS_DIR),
        name="egg_candling",

        # Reproducibility
        seed=RANDOM_SEED,

        # Stop early if validation stops improving
        patience=10,

        # Windows compatibility
        workers=0,

        # Automatically use available GPU if possible
        device=0 if __import__("torch").cuda.is_available() else "cpu"
    )

    return model, results


def save_best_model():

    print("\n==============================================")
    print(" SAVING TRAINED MODEL")
    print("==============================================\n")

    best_model = RUNS_DIR / "egg_candling" / "weights" / "best.pt"

    if not best_model.exists():
        raise FileNotFoundError(
            f"best.pt was not found:\n{best_model}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy best model
    shutil.copy2(best_model, FINAL_MODEL)

    print(f"Trained model saved to:")
    print(FINAL_MODEL)

    # Save class names
    classes_file = MODEL_DIR / "classes.txt"

    with open(classes_file, "w", encoding="utf-8") as file:
        for stage in STAGES:
            file.write(stage + "\n")

    print(f"\nClass names saved to:")
    print(classes_file)


def test_model():

    print("\n==============================================")
    print(" TESTING TRAINED MODEL")
    print("==============================================\n")

    if not FINAL_MODEL.exists():
        print("ERROR: Trained model does not exist.")
        return

    print("Loading trained model...")

    model = YOLO(str(FINAL_MODEL))

    print("\nModel loaded successfully!")

    print("\nClasses detected by model:")

    for index, name in model.names.items():
        print(f"{index}: {name}")

    print("\n==============================================")
    print(" TRAINING COMPLETE!")
    print("==============================================")

    print("\nYour trained AI model is:")
    print(FINAL_MODEL)

    print("\nNext step:")
    print("Connect this model to app.py")
    print("so the camera can automatically predict Day 1-21.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("======================================================")
    print("     CHICKEN EGG CANDLING AI TRAINING SYSTEM")
    print("======================================================")
    print("Classes: Day 1 - Day 21")
    print("Model: YOLO Classification")
    print("======================================================")

    try:

        # Step 1
        train_dir, val_dir = prepare_dataset()

        # Step 2
        train_model(train_dir, val_dir)

        # Step 3
        save_best_model()

        # Step 4
        test_model()

    except Exception as error:

        print("\n")
        print("======================================================")
        print(" TRAINING ERROR")
        print("======================================================")
        print(error)
        print("\nPlease check the error above.")
        print("======================================================")


if __name__ == "__main__":
    main()