"""
Configurable incubation-day → development-stage labels.

The YOLO model only predicts an incubation day (Day_1 ... Day_21).
These strings are display labels only — they are not produced by the model.
Edit this mapping if a poultry specialist provides better descriptions.
"""

from config import class_folder_from_display, display_day_name

# Easy to edit. Keys may be "Day_1" or "Day 1".
STAGE_MAP = {
    "Day_1": "Early Development",
    "Day_2": "Early Development",
    "Day_3": "Early Development",
    "Day_4": "Early Development",
    "Day_5": "Early Vascular Development",
    "Day_6": "Early Vascular Development",
    "Day_7": "Embryonic Development",
    "Day_8": "Embryonic Development",
    "Day_9": "Embryonic Development",
    "Day_10": "Embryonic Development",
    "Day_11": "Mid Embryonic Development",
    "Day_12": "Mid Embryonic Development",
    "Day_13": "Mid Embryonic Development",
    "Day_14": "Mid Embryonic Development",
    "Day_15": "Late Embryonic Development",
    "Day_16": "Late Embryonic Development",
    "Day_17": "Late Embryonic Development",
    "Day_18": "Pre-Hatch Development",
    "Day_19": "Pre-Hatch Development",
    "Day_20": "Hatch Window",
    "Day_21": "Hatch Window",
}

DEFAULT_STAGE = "Incubation Day Class"


def get_development_stage(day_label: str) -> str:
    if not day_label:
        return DEFAULT_STAGE
    folder = class_folder_from_display(day_label)
    if folder in STAGE_MAP:
        return STAGE_MAP[folder]
    pretty = display_day_name(day_label)
    if pretty in STAGE_MAP:
        return STAGE_MAP[pretty]
    return DEFAULT_STAGE
