"""
Phase 1: Landmark Extraction
Extracts body and hand landmarks from a video clip using MediaPipe.
Outputs one CSV per clip to data/landmarks/.

Usage:
    python extract_landmarks.py --video data/clips/teacher_tatkaar_01.mp4
    python extract_landmarks.py --video data/clips/ --batch   # process all clips
"""

import argparse
import csv
import os
import cv2
import mediapipe as mp
import numpy as np

VISIBILITY_THRESHOLD = 0.5

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

BODY_LANDMARK_NAMES = [lm.name.lower() for lm in mp_pose.PoseLandmark]


def extract_landmarks(video_path: str, output_dir: str = "data/landmarks") -> str:
    os.makedirs(output_dir, exist_ok=True)

    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{clip_name}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {clip_name}: {total_frames} frames at {fps:.1f} fps")

    rows = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            hand_result = hands.process(rgb)

            row = {
                "frame": frame_idx,
                "timestamp_ms": int((frame_idx / fps) * 1000),
            }

            # Body landmarks (33 keypoints)
            if pose_result.pose_landmarks:
                for i, lm in enumerate(pose_result.pose_landmarks.landmark):
                    name = BODY_LANDMARK_NAMES[i]
                    # Always store visibility; only store position if confident
                    row[f"body_{name}_vis"] = round(lm.visibility, 4)
                    if lm.visibility >= VISIBILITY_THRESHOLD:
                        row[f"body_{name}_x"] = round(lm.x, 6)
                        row[f"body_{name}_y"] = round(lm.y, 6)
                        row[f"body_{name}_z"] = round(lm.z, 6)
                    else:
                        row[f"body_{name}_x"] = None
                        row[f"body_{name}_y"] = None
                        row[f"body_{name}_z"] = None

            # Hand landmarks (21 keypoints per hand)
            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for hand_lm, handedness in zip(
                    hand_result.multi_hand_landmarks, hand_result.multi_handedness
                ):
                    hand_label = handedness.classification[0].label.lower()  # "left" or "right"
                    for j, lm in enumerate(hand_lm.landmark):
                        row[f"hand_{hand_label}_{j}_x"] = round(lm.x, 6)
                        row[f"hand_{hand_label}_{j}_y"] = round(lm.y, 6)
                        row[f"hand_{hand_label}_{j}_z"] = round(lm.z, 6)

            rows.append(row)

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}", end="\r")

            frame_idx += 1

    cap.release()

    if not rows:
        print(f"Warning: no frames extracted from {video_path}")
        return output_path

    # Write CSV with consistent columns across all rows
    all_keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nSaved {len(rows)} frames to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract pose and hand landmarks from video")
    parser.add_argument("--video", required=True, help="Path to video file or directory (with --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all .mp4 files in directory")
    parser.add_argument("--output-dir", default="data/landmarks", help="Output directory for CSVs")
    args = parser.parse_args()

    if args.batch:
        video_dir = args.video
        videos = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".mov", ".avi"))
        ]
        print(f"Found {len(videos)} videos in {video_dir}")
        for v in videos:
            extract_landmarks(v, args.output_dir)
    else:
        extract_landmarks(args.video, args.output_dir)


if __name__ == "__main__":
    main()
