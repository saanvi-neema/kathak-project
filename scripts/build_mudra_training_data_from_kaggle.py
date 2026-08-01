"""
Builds a labeled mudra training dataset from the "Asanyukta Kathak Mudra"
Kaggle dataset (demon2angel/asanyukta-kathak-mudra, CC BY 4.0 via Roboflow)
-- 1,560 real Kathak mudra photos, 26 classes. Unlike
build_mudra_training_data_from_images.py's source (Bharatanatyam, a
different though related dance form), this is genuinely in-domain data,
built specifically to close the domain gap found by testing the
Bharatanatyam-only classifier against this project's own real Kathak
footage (see methods.md).

Dataset format is YOLO object-detection (train/valid/test each with an
images/ and labels/ folder; each label file is "<class_idx> <x> <y> <w>
<h>", class names in data.yaml) rather than one folder per class -- this
script reads the class index from each label file instead of the folder
name, everything else matches build_mudra_training_data_from_images.py's
approach (MediaPipe hand detection in IMAGE mode, same output schema so
mudra_classifier.py doesn't care which script produced its input).

KATHAK_LABEL_MAP is hand-checked against mudra_reference.MUDRA_RULES, same
as the Bharatanatyam script's mapping. "katak" is mapped to katakamukh_1 as
the closest single representative -- this dataset doesn't distinguish the 3
katakamukh variants the way the Bharatanatyam one does.

Usage:
    python build_mudra_training_data_from_kaggle.py
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import pandas as pd
import yaml
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from extract_landmarks import HAND_MODEL_PATH  # noqa: E402

# Dataset class name (from data.yaml) -> this project's mudra name.
# Cross-checked by hand against mudra_reference.MUDRA_RULES.
KATHAK_LABEL_MAP = {
    "Hamsapaksha": "hanspaksha",
    "aral": "aral",
    "ardhachandra": "ardhachandra",
    "ardhpataka": "ardhapataka",
    "bhramara": "bhramara",
    "chandrakala": "chandrakala",
    "chatur": "chatur",
    "hansaasya": "hansasya",
    "kangul": "kangul",
    "kapitth": "kapitth",
    "kartarimukh": "kartarimukh",
    "katak": "katakamukh_1",  # closest single representative -- dataset doesn't split variants 1/2/3
    "mayur": "mayur",
    "mrighasheesh": "mrigasheesh",
    "mukul": "mukul",
    "mushti": "mushti",
    "padamkosh": "padmakosh",
    "pataka": "pataka",
    "sarpsheesh": "sarpasheesh",
    "shikhar": "shikhar",
    "shuktund": "shuktund",
    "sinhamukh": "singhamukh",
    "soochi": "soochi",
    "tamrachud": "tamrachud",
    "tripataka": "tripataka",
    "trishool": "trishool",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def make_image_hand_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,  # see build_mudra_training_data_from_images.py -- measured directly against the Bharatanatyam set, applied here too since these are similarly-framed studio-ish photos
        min_hand_presence_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_one_image(landmarker, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks or not result.handedness:
        return None, None
    side = result.handedness[0][0].category_name.lower()
    points = {i: (p.x, p.y) for i, p in enumerate(result.hand_landmarks[0])}
    return side, points


def read_label_class(label_path):
    """First token of a YOLO label file is the class index. Returns None if the file is missing/empty."""
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        first_line = f.readline().strip()
    if not first_line:
        return None
    return int(first_line.split()[0])


def build_dataset(raw_dir, output_csv, max_per_class=None):
    yaml_path = os.path.join(raw_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        print(f"data.yaml not found in {raw_dir}")
        return
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    class_names = config["names"]

    landmarker = make_image_hand_landmarker()
    counts = {name: 0 for name in KATHAK_LABEL_MAP.values()}
    rows = []
    try:
        for split in ["train", "valid", "test"]:
            images_dir = os.path.join(raw_dir, split, "images")
            labels_dir = os.path.join(raw_dir, split, "labels")
            if not os.path.isdir(images_dir):
                continue

            for fname in sorted(os.listdir(images_dir)):
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                stem = os.path.splitext(fname)[0]
                label_path = os.path.join(labels_dir, stem + ".txt")
                class_idx = read_label_class(label_path)
                if class_idx is None:
                    continue
                dataset_name = class_names[class_idx]
                mudra_name = KATHAK_LABEL_MAP.get(dataset_name)
                if mudra_name is None:
                    continue
                if max_per_class and counts[mudra_name] >= max_per_class:
                    continue

                side, points = extract_one_image(landmarker, os.path.join(images_dir, fname))
                if side is None:
                    continue

                row = {f"hand_{side}_{i}_x": x for i, (x, y) in points.items()}
                row.update({f"hand_{side}_{i}_y": y for i, (x, y) in points.items()})
                row["mudra"] = mudra_name
                row["hand_side"] = side
                rows.append(row)
                counts[mudra_name] += 1
    finally:
        landmarker.close()

    print("Per-mudra usable image counts:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    if not rows:
        print("No training rows produced.")
        return

    dataset = pd.DataFrame(rows)
    dataset["timestamp_ms"] = range(len(dataset))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dataset.to_csv(output_csv, index=False)
    print(f"\nSaved {len(dataset)} labeled images across {dataset['mudra'].nunique()} mudras to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Build a labeled mudra dataset from the Kaggle Kathak dataset")
    parser.add_argument("--raw-dir", default="data/mudra_training/external/kathak_kaggle")
    parser.add_argument("--output", default="data/mudra_training/dataset_kathak.csv")
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.output, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
