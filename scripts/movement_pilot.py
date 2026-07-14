"""
Experiment 1 pilot: does pose tracking survive ordinary (non-spin) Kathak-ish
movement? Compares MediaPipe Pose vs MoveNet Thunder.

For each clip in CLIPS below:
  1. Run MediaPipe Pose per-frame (Tasks API) and MoveNet Thunder per-frame
     (TFLite), recording keypoints + confidence.
  2. Save an overlay video per model (skeleton drawn on frame) for QA.
  3. Save every Nth frame as a still image (both models side by side) into
     a review folder, for manual Good/Usable/Failed labeling.
  4. Print a per-model detection-rate summary (mean confidence, % frames
     with a person detected).

This does NOT do frame-level Good/Usable/Failed labeling automatically --
that step is manual (or done by visually reviewing the sampled stills /
overlay videos). This script just gets the data and overlays ready for it.

Usage:
    python movement_pilot.py
"""

import csv
import os

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MEDIAPIPE_MODEL_PATH = "models/pose_landmarker_full.task"
MOVENET_MODEL_PATH = "models/movenet_thunder.tflite"
MOVENET_INPUT_SIZE = 256  # thunder variant expects 256x256

RAW_DIR = "data/raw/movement"
OUT_DIR = "data/landmarks/movement_pilot"

SAMPLE_EVERY_N_FRAMES = 15  # ~2 stills/sec at 30fps, for manual QA review

CLIPS = [
    "movement_01.mov",
]

# BlazePose (MediaPipe) 33-point connections, subset used for drawing.
MEDIAPIPE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# MoveNet 17-keypoint COCO order + skeleton connections.
MOVENET_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
MOVENET_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

CONF_THRESHOLD = 0.3


def make_mediapipe_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def make_movenet_interpreter():
    interpreter = tf.lite.Interpreter(model_path=MOVENET_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def run_movenet(interpreter, frame_rgb):
    h, w = frame_rgb.shape[:2]
    img = cv2.resize(frame_rgb, (MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE))
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]["index"])[0][0]  # (17, 3): y, x, score
    return keypoints  # normalized [0,1] y, x, score


def draw_mediapipe(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in MEDIAPIPE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        vis = landmarks[i].visibility
        color = (0, 255, 0) if vis >= 0.5 else (0, 0, 255)
        cv2.circle(frame, (x, y), 4, color, -1)


def draw_movenet(frame, keypoints, w, h):
    pts = [(int(kp[1] * w), int(kp[0] * h)) for kp in keypoints]
    scores = [kp[2] for kp in keypoints]
    for a, b in MOVENET_CONNECTIONS:
        if scores[a] >= CONF_THRESHOLD and scores[b] >= CONF_THRESHOLD:
            cv2.line(frame, pts[a], pts[b], (255, 128, 0), 2)
    for i, (x, y) in enumerate(pts):
        color = (255, 128, 0) if scores[i] >= CONF_THRESHOLD else (0, 0, 255)
        cv2.circle(frame, (x, y), 4, color, -1)


def process_video(video_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stills_dir = os.path.join(out_dir, "review_stills")
    os.makedirs(stills_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_overlay_path = os.path.join(out_dir, f"{clip_name}_mediapipe_overlay.mp4")
    mn_overlay_path = os.path.join(out_dir, f"{clip_name}_movenet_overlay.mp4")
    mp_writer = cv2.VideoWriter(mp_overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    mn_writer = cv2.VideoWriter(mn_overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    landmarker = make_mediapipe_landmarker()
    movenet = make_movenet_interpreter()

    rows = []
    mp_frames_with_pose = 0
    mn_frames_with_pose = 0
    mp_confidences = []
    mn_confidences = []

    try:
        frame_idx = 0
        last_ts = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- MediaPipe ---
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            if ts_ms <= last_ts:
                ts_ms = last_ts + 1
            last_ts = ts_ms
            mp_result = landmarker.detect_for_video(mp_image, ts_ms)

            mp_frame = frame.copy()
            has_mp_pose = bool(mp_result.pose_landmarks)
            row = {"frame": frame_idx, "timestamp_ms": ts_ms, "mp_pose_detected": has_mp_pose}
            if has_mp_pose:
                lm = mp_result.pose_landmarks[0]
                mp_frames_with_pose += 1
                mean_vis = float(np.mean([p.visibility for p in lm]))
                mp_confidences.append(mean_vis)
                row["mp_mean_visibility"] = mean_vis
                draw_mediapipe(mp_frame, lm, w, h)
            else:
                row["mp_mean_visibility"] = None
            mp_writer.write(mp_frame)

            # --- MoveNet ---
            mn_frame = frame.copy()
            keypoints = run_movenet(movenet, rgb)
            mean_score = float(np.mean(keypoints[:, 2]))
            has_mn_pose = mean_score >= CONF_THRESHOLD
            row["mn_pose_detected"] = has_mn_pose
            row["mn_mean_score"] = mean_score
            if has_mn_pose:
                mn_frames_with_pose += 1
            mn_confidences.append(mean_score)
            draw_movenet(mn_frame, keypoints, w, h)
            mn_writer.write(mn_frame)

            # --- Save side-by-side still every N frames for manual QA ---
            if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
                side_by_side = np.hstack([mp_frame, mn_frame])
                still_path = os.path.join(stills_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(still_path, side_by_side)

            rows.append(row)
            frame_idx += 1
    finally:
        landmarker.close()

    cap.release()
    mp_writer.release()
    mn_writer.release()

    csv_path = os.path.join(out_dir, f"{clip_name}_detections.csv")
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    return {
        "clip": clip_name,
        "total_frames": total_frames,
        "mp_frames_with_pose": mp_frames_with_pose,
        "mp_mean_confidence": float(np.mean(mp_confidences)) if mp_confidences else 0.0,
        "mn_frames_with_pose": mn_frames_with_pose,
        "mn_mean_confidence": float(np.mean(mn_confidences)) if mn_confidences else 0.0,
        "mp_overlay_path": mp_overlay_path,
        "mn_overlay_path": mn_overlay_path,
        "stills_dir": stills_dir,
        "csv_path": csv_path,
    }


def main():
    results = []
    for filename in CLIPS:
        video_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(video_path):
            print(f"SKIP (not found): {video_path}")
            continue
        clip_name = os.path.splitext(filename)[0]
        out_dir = os.path.join(OUT_DIR, clip_name)
        print(f"Processing {filename}...")
        r = process_video(video_path, out_dir)
        results.append(r)

    print("\n=== Detection quality (frame-level, no manual QA yet) ===")
    print(f"{'clip':<16}{'mp_detect%':>12}{'mp_conf':>10}{'mn_detect%':>12}{'mn_conf':>10}")
    for r in results:
        mp_pct = 100 * r["mp_frames_with_pose"] / r["total_frames"]
        mn_pct = 100 * r["mn_frames_with_pose"] / r["total_frames"]
        print(
            f"{r['clip']:<16}{mp_pct:>11.1f}%{r['mp_mean_confidence']:>10.2f}"
            f"{mn_pct:>11.1f}%{r['mn_mean_confidence']:>10.2f}"
        )

    print("\nOutputs written under:", OUT_DIR)
    print("Review *_mediapipe_overlay.mp4 / *_movenet_overlay.mp4 and the")
    print("review_stills/ side-by-side frames to manually label Good/Usable/Failed.")


if __name__ == "__main__":
    main()
