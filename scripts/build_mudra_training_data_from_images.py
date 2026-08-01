"""
Builds a labeled mudra training dataset from a folder of static per-mudra
photos, instead of build_mudra_training_data.py's video-clip approach.

Source: jisharajr/Bharatanatyam-Mudra-Dataset (GitHub, CC-BY-SA 4.0) -- 15
volunteers' worth of studio-shot single-hand mudra photos. Bharatanatyam and
Kathak's asamyukta hastas come from the same Natya Shastra vocabulary, so
mudra names/shapes match this project's own mudra_reference.py, but this
hasn't been cross-checked against real Kathak footage for styling
differences -- worth revisiting once real Kathak clips exist.

FOLDER_NAME_MAP below is an explicit, hand-verified mapping from the
dataset's folder names to this project's mudra names -- built by checking
the dataset's own README table against mudra_reference.MUDRA_RULES, not a
fuzzy/automatic match, so a labeling mistake here can't silently happen.
Only folders with a confirmed mapping are used; the dataset's double-hand
(samyukta hasta) folders and anything not in our 28-mudra list are skipped
outright. "sandansh" has no match in this dataset and needs sourcing
separately (real footage or another dataset).

Since these are static photos, not video, there's no hold-vs-transition
motion detection needed (unlike build_mudra_training_data.py) -- every
image already is one clean, single held pose. Output schema matches
build_mudra_training_data.py's exactly (hand_<side>_<i>_x/y + "mudra" +
"hand_side" columns), so mudra_classifier.py works unchanged regardless of
which script produced the dataset.

Usage:
    python build_mudra_training_data_from_images.py
    python build_mudra_training_data_from_images.py --max-per-class 150
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from extract_landmarks import HAND_MODEL_PATH  # noqa: E402

# Dataset folder name -> this project's mudra name. Cross-checked by hand
# against the dataset README's single-hand (Asamyukta Hasta) table and
# mudra_reference.MUDRA_RULES -- anything not listed here is deliberately
# skipped (double-hand folders, or names with no confirmed match).
FOLDER_NAME_MAP = {
    "Pathaka(1)": "pataka",
    "Tripathaka(1)": "tripataka",
    "Ardhapathaka(1)": "ardhapataka",
    "Mayura(1)": "mayur",
    "Katrimukha(1)": "kartarimukh",
    "Ardhachandran(1)": "ardhachandra",
    "Aralam(1)": "aral",
    "Shukatundam(1)": "shuktund",
    "Mushti(1)": "mushti",
    "Sikharam(1)": "shikhar",
    "Kapith(1)": "kapitth",
    "Katakamukha_1": "katakamukh_1",
    "Katakamukha_2": "katakamukh_2",
    "Katakamukha_3": "katakamukh_3",
    "Suchi(1)": "soochi",
    "Chandrakala(1)": "chandrakala",
    "Padmakosha(1)": "padmakosh",
    "Sarpasirsha(1)": "sarpasheesh",
    "Mrigasirsha(1)": "mrigasheesh",
    "Simhamukham(1)": "singhamukh",
    "Kangulam(1)": "kangul",
    "Alapadmam(1)": "alapadma",
    "Mukulam(1)": "mukul",
    "Chaturam(1)": "chatur",
    "Bramaram(1)": "bhramara",
    "Hamsasyam(1)": "hansasya",
    "Hamsapaksha(1)": "hanspaksha",
    "Tamarachudam(1)": "tamrachud",
    "Trishulam(1)": "trishool",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def make_image_hand_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,  # these are single-hand (asamyukta) photos
        # Lower than the video-pipeline default (0.5) -- measured directly
        # against this dataset: 0.5 only detected a hand in ~45% of images
        # (tight studio crops, unusual hand angles the video-trained
        # detector wasn't tuned for), 0.3 recovered ~60% without producing
        # degenerate/garbage detections (checked landmark spread directly).
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_one_image(landmarker, image_path):
    """Returns (hand_side, {landmark_index: (x, y)}) or (None, None) if no hand was detected."""
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


def build_dataset(raw_dir, output_csv, max_per_class=None):
    if not os.path.isdir(raw_dir):
        print(f"Raw dataset directory not found: {raw_dir}")
        return

    landmarker = make_image_hand_landmarker()
    rows = []
    try:
        for folder_name, mudra_name in FOLDER_NAME_MAP.items():
            folder_path = os.path.join(raw_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"  WARNING: expected folder not found, skipping: {folder_name}")
                continue

            images = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS))

            found = 0
            scanned = 0
            for fname in images:
                scanned += 1
                side, points = extract_one_image(landmarker, os.path.join(folder_path, fname))
                if side is None:
                    continue
                row = {f"hand_{side}_{i}_x": x for i, (x, y) in points.items()}
                row.update({f"hand_{side}_{i}_y": y for i, (x, y) in points.items()})
                row["mudra"] = mudra_name
                row["hand_side"] = side
                rows.append(row)
                found += 1
                if max_per_class and found >= max_per_class:
                    break

            print(f"{folder_name} -> '{mudra_name}': {found} usable images (scanned {scanned}/{len(images)})")
    finally:
        landmarker.close()

    if not rows:
        print("No training rows produced.")
        return

    dataset = pd.DataFrame(rows)
    dataset["timestamp_ms"] = range(len(dataset))  # not real timestamps, just keeps the schema consistent
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dataset.to_csv(output_csv, index=False)
    print(f"\nSaved {len(dataset)} labeled images across {dataset['mudra'].nunique()} mudras to {output_csv}")

    missing = set(FOLDER_NAME_MAP.values()) - set(dataset["mudra"].unique())
    if missing:
        print(f"Mudras with zero usable images: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser(description="Build a labeled mudra dataset from the external Bharatanatyam image dataset")
    parser.add_argument("--raw-dir", default="data/mudra_training/external/bharatanatyam")
    parser.add_argument("--output", default="data/mudra_training/dataset.csv")
    parser.add_argument("--max-per-class", type=int, default=150,
                         help="Cap images processed per mudra -- keeps runtime bounded; 150 is still 6-19x more per-class data than the manual-recording plan this replaced.")
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.output, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
