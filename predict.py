"""
Run YOLO classification with the locally fine-tuned egg candling model.
Predictions always come from models/egg_candling_best.pt — never hardcoded.
"""

from __future__ import annotations

from pathlib import Path

from config import MODEL_PATH, MIN_CONFIDENCE, display_day_name
from stage_mapping import get_development_stage

MODEL_MISSING_MESSAGE = (
    "AI model not trained yet. Please train the model first."
)

_model = None
_model_error = None


class ModelNotTrainedError(RuntimeError):
    pass


def model_is_ready() -> bool:
    return MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 0


def load_model():
    global _model, _model_error
    if not model_is_ready():
        _model = None
        raise ModelNotTrainedError(MODEL_MISSING_MESSAGE)

    if _model is not None:
        return _model

    from ultralytics import YOLO

    try:
        _model = YOLO(str(MODEL_PATH))
        _model_error = None
        return _model
    except Exception as error:
        _model_error = str(error)
        raise RuntimeError(f"Failed to load trained model at {MODEL_PATH}: {error}") from error


def _top_predictions(result, limit: int = 3) -> list[dict]:
    probs = getattr(result, "probs", None)
    if probs is None:
        raise RuntimeError(
            "The loaded model did not return classification probabilities. "
            "Train a YOLO classification model (not a detection-only model)."
        )

    names = result.names
    data = probs.data
    try:
        values = data.cpu().numpy()
    except Exception:
        values = data.numpy() if hasattr(data, "numpy") else list(data)

    ranked = sorted(enumerate(values), key=lambda item: float(item[1]), reverse=True)
    top = []
    for index, score in ranked[:limit]:
        class_name = names.get(index, str(index)) if isinstance(names, dict) else names[index]
        top.append(
            {
                "class": str(class_name),
                "day": display_day_name(str(class_name)),
                "confidence": round(float(score) * 100.0, 2),
                "confidence_raw": float(score),
            }
        )
    return top


def predict_image(image_path: str | Path, min_confidence: float = MIN_CONFIDENCE) -> dict:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    model = load_model()
    results = model.predict(source=str(path), verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no result for this image.")

    result = results[0]
    top = _top_predictions(result, limit=3)
    if not top:
        raise RuntimeError("YOLO returned empty classification scores.")

    best = top[0]
    confidence_raw = best["confidence_raw"]
    low_confidence = confidence_raw < min_confidence

    if low_confidence:
        predicted_day = None
        development_stage = None
        status = (
            "Low confidence — please recapture the egg image under better candling conditions."
        )
    else:
        predicted_day = best["day"]
        development_stage = get_development_stage(best["class"])
        status = "Prediction successful"

    return {
        "success": True,
        "predicted_class": best["class"],
        "predicted_day": predicted_day,
        "raw_predicted_day": best["day"],
        "development_stage": development_stage,
        "confidence": best["confidence"],
        "confidence_raw": confidence_raw,
        "low_confidence": low_confidence,
        "min_confidence": min_confidence,
        "status": status,
        "top_predictions": [
            {"day": item["day"], "confidence": item["confidence"]}
            for item in top
        ],
        "model_path": str(MODEL_PATH),
    }


def format_cli_output(prediction: dict) -> str:
    day = prediction.get("predicted_day") or prediction.get("raw_predicted_day") or "Uncertain"
    stage = prediction.get("development_stage") or "Not assigned (low confidence)"
    lines = [
        f"Predicted Day: {day}",
        f"Development Stage: {stage}",
        f"Confidence: {prediction['confidence']:.2f}%",
        f"Status: {prediction['status']}",
        "Top predictions:",
    ]
    for item in prediction["top_predictions"]:
        lines.append(f"  - {item['day']}: {item['confidence']:.2f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict incubation day from a candled egg image.")
    parser.add_argument("image", help="Path to an egg image")
    args = parser.parse_args()

    if not model_is_ready():
        print(MODEL_MISSING_MESSAGE)
        print(f"Expected model file: {MODEL_PATH}")
        print("Train with:")
        print("  py prepare_dataset.py")
        print("  py train_model.py")
        raise SystemExit(1)

    output = predict_image(args.image)
    print(format_cli_output(output))
    print()
    print(json.dumps(output, indent=2))
