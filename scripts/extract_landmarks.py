"""
Phase 1: Landmark Extraction
Extracts body and hand landmarks from a video clip using MediaPipe's Tasks API.

Uses PoseLandmarker + HandLandmarker + FaceLandmarker (not the old
mp.solutions API, which mediapipe 0.10.35 removed on Windows -- see the
pilot scripts for the same workaround). Requires downloaded model bundles:
    models/pose_landmarker_full.task
    models/hand_landmarker.task
    models/face_landmarker.task

Outputs one CSV per clip to data/landmarks/.

Usage:
    python extract_landmarks.py --video data/clips/teacher_tatkaar_01.mp4
    python extract_landmarks.py --video data/clips/ --batch   # process all clips
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VISIBILITY_THRESHOLD = 0.5

# Real bottleneck found and fixed: extract_frame_landmarks() used to run the
# pose/hand/face models back-to-back on every frame, so a frame's total cost
# was the SUM of all three. They're independent (same input frame, no shared
# state), and MediaPipe's Tasks API calls are C++ inference that release the
# GIL while running, so submitting all three at once lets them actually run
# concurrently -- a frame's cost becomes closer to the SLOWEST of the three,
# not their sum. Module-level and reused across calls rather than created
# per frame (thread creation isn't free, and this runs once per frame across
# thousands of frames).
_LANDMARKER_EXECUTOR = ThreadPoolExecutor(max_workers=3)

POSE_MODEL_PATH = "models/pose_landmarker_full.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"
FACE_MODEL_PATH = "models/face_landmarker.task"

# BlazePose 33-point topology, in landmark-index order. Same order as the old
# mp.solutions.pose.PoseLandmark enum -- the Tasks API kept the same indices,
# just dropped the enum.
BODY_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def make_pose_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def make_hand_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def make_face_landmarker():
    """
    output_face_blendshapes=True is the point of running this at all --
    the 52 blendshape scores (mouthSmileLeft, browDownLeft, eyeWideRight,
    jawOpen, noseSneerLeft, etc.) are semantically meaningful facial-muscle-
    movement measurements, much closer to rasa_reference.py's expression
    criteria than raw face landmark positions would be.
    """
    base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def next_video_timestamp_ms(frame_idx, fps, last_ts):
    """
    Monotonic-guarded ts_ms rule extract_landmarks() has always used, factored
    out so a live capture session (see app/live_pipeline.py) can keep counting
    seamlessly across many separate short video chunks instead of resetting
    to 0 at the start of each one.
    """
    ts_ms = int((frame_idx / fps) * 1000)
    if ts_ms <= last_ts:
        ts_ms = last_ts + 1
    return ts_ms


def extract_frame_landmarks(pose_landmarker, hand_landmarker, face_landmarker, frame_bgr, frame_idx, ts_ms):
    """
    Runs all three landmarkers on one already-decoded BGR frame and returns
    one row dict in extract_landmarks()'s CSV row schema. Pure function of
    its inputs (no other state) -- factored out of extract_landmarks()'s loop
    body so a live capture session can reuse the same long-lived landmarker
    instances across many chunks instead of recreating them (and reloading
    the .task model bundles) every chunk. extract_landmarks() itself calls
    this unchanged, so its own output is unaffected by this split.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pose_future = _LANDMARKER_EXECUTOR.submit(pose_landmarker.detect_for_video, mp_image, ts_ms)
    hand_future = _LANDMARKER_EXECUTOR.submit(hand_landmarker.detect_for_video, mp_image, ts_ms)
    face_future = _LANDMARKER_EXECUTOR.submit(face_landmarker.detect_for_video, mp_image, ts_ms)
    pose_result = pose_future.result()
    hand_result = hand_future.result()
    face_result = face_future.result()

    row = {
        "frame": frame_idx,
        "timestamp_ms": ts_ms,
    }

    # Body landmarks (33 keypoints)
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]
        for i, name in enumerate(BODY_LANDMARK_NAMES):
            p = lm[i]
            # Always store visibility; only store position if confident
            row[f"body_{name}_vis"] = round(p.visibility, 4)
            if p.visibility >= VISIBILITY_THRESHOLD:
                row[f"body_{name}_x"] = round(p.x, 6)
                row[f"body_{name}_y"] = round(p.y, 6)
                row[f"body_{name}_z"] = round(p.z, 6)
            else:
                row[f"body_{name}_x"] = None
                row[f"body_{name}_y"] = None
                row[f"body_{name}_z"] = None

    # Hand landmarks (21 keypoints per hand)
    if hand_result.hand_landmarks and hand_result.handedness:
        for hand_lm, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            # MediaPipe's handedness classifier assumes a mirrored/selfie-
            # style input image and is documented as needing the output
            # swapped otherwise. Every real source this app captures from is
            # non-mirrored: phone camera apps save the corrected (non-
            # mirrored) file by default even from the front camera (only an
            # explicit "mirror front camera" setting changes that), and the
            # browser's getUserMedia/MediaRecorder in live mode captures the
            # raw, unmirrored sensor feed regardless of any CSS mirroring
            # applied to the on-screen preview for the dancer's convenience.
            # So the raw classification is flipped here rather than trusted
            # as-is. This only affects the hand_side label attached to a
            # detected mudra event (which physical hand gets reported) --
            # predict_mudra() strips this prefix before scoring (see
            # app/pipeline.py's run_mudra_analysis), so mudra
            # identification/scoring itself is completely unaffected.
            raw_label = handedness[0].category_name.lower()  # "left" or "right", per MediaPipe's mirrored convention
            hand_label = "right" if raw_label == "left" else "left"
            for j, p in enumerate(hand_lm):
                row[f"hand_{hand_label}_{j}_x"] = round(p.x, 6)
                row[f"hand_{hand_label}_{j}_y"] = round(p.y, 6)
                row[f"hand_{hand_label}_{j}_z"] = round(p.z, 6)

    # Face blendshapes (52 named scores) -- see rasa_reference.py for
    # how these map to navarasa expressions.
    if face_result.face_blendshapes:
        for bs in face_result.face_blendshapes[0]:
            row[f"face_{bs.category_name}"] = round(bs.score, 4)

    return row


def extract_landmarks(video_path: str, output_dir: str = "data/landmarks", frame_stride: int = 1) -> str:
    """
    frame_stride: only run the real (expensive) 3-model detection on every
    Nth frame; frames in between hold the last detected frame's landmark
    values (own frame/timestamp_ms still recorded) instead of re-running
    detection. Real, disclosed speed/resolution trade-off, not a silent
    approximation -- events chakkar/mudra/timing care about span many
    frames, so holding position for 1-2 intermediate frames out of every N
    has a small effect on that data's smoothness, not on whether an event
    gets detected at all. Default 1 (every frame, unchanged) -- the batch
    upload path (analyze_video) is the only caller that raises this.
    """
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

    pose_landmarker = make_pose_landmarker()
    hand_landmarker = make_hand_landmarker()
    face_landmarker = make_face_landmarker()
    try:
        frame_idx = 0
        last_ts = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts_ms = next_video_timestamp_ms(frame_idx, fps, last_ts)
            last_ts = ts_ms

            if frame_stride <= 1 or frame_idx % frame_stride == 0:
                row = extract_frame_landmarks(pose_landmarker, hand_landmarker, face_landmarker, frame, frame_idx, ts_ms)
                last_row = row
            else:
                row = dict(last_row)
                row["frame"] = frame_idx
                row["timestamp_ms"] = ts_ms
            rows.append(row)

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}", end="\r")

            frame_idx += 1
    finally:
        pose_landmarker.close()
        hand_landmarker.close()
        face_landmarker.close()

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
    parser.add_argument("--batch", action="store_true", help="Process all video files in directory")
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
