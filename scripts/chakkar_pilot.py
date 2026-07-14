"""
Chakkar pilot experiment (NOT part of the Phase 1-7 pipeline).

Question this answers: does MediaPipe Pose survive a Kathak chakkar, and can a
naive shoulder-line orientation signal (atan2) count full rotations?

For each clip in CLIPS below:
  1. Run MediaPipe Pose per-frame, record shoulder/hip/nose landmarks + visibility.
  2. Save an overlay video (skeleton drawn on frame) for manual tracking-quality QA.
  3. Compute shoulder-line orientation angle per frame, unwrap it across the whole
     clip, and estimate rotation count from total unwrapped rotation / 360.
  4. Save a CSV of per-frame angle data and a PNG plot (raw wrapped angle +
     unwrapped angle vs frame).
  5. Print a detection-quality summary and the estimated vs ground-truth count.

Usage:
    python chakkar_pilot.py
"""

import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VISIBILITY_THRESHOLD = 0.5
MODEL_PATH = "models/pose_landmarker_full.task"

RAW_DIR = "data/raw"
OUT_DIR = "data/landmarks/chakkar_pilot"

# clip filename -> human-annotated ground-truth chakkar count
CLIPS = {
    "chakkar_01.mov": 1,
    "chakkar_02.mov": 3,
    "chakkar_03.mov": 10,
}

# BlazePose 33-point topology (same indices in both the old "solutions" API
# and the new Tasks API).
LM_NOSE = 0
LM_L_SHOULDER, LM_R_SHOULDER = 11, 12
LM_L_HIP, LM_R_HIP = 23, 24

# Minimal skeleton connections for overlay QA drawing (subset of the standard
# BlazePose POSE_CONNECTIONS from the old mp.solutions.drawing_utils).
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


def make_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def draw_landmarks(frame, landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        vis = landmarks[i].visibility
        color = (0, 255, 0) if vis >= VISIBILITY_THRESHOLD else (0, 0, 255)
        cv2.circle(frame, (x, y), 4, color, -1)


def process_video(video_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlay_path = os.path.join(out_dir, f"{clip_name}_overlay.mp4")
    writer = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    frames_with_pose = 0
    frames_with_both_shoulders = 0

    landmarker = make_landmarker()
    try:
        frame_idx = 0
        last_ts = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            if ts_ms <= last_ts:
                ts_ms = last_ts + 1
            last_ts = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)

            row = {"frame": frame_idx, "timestamp_ms": ts_ms}
            has_pose = bool(result.pose_landmarks)
            row["pose_detected"] = has_pose

            if has_pose:
                frames_with_pose += 1
                lm = result.pose_landmarks[0]

                for name, idx in [
                    ("l_shoulder", LM_L_SHOULDER),
                    ("r_shoulder", LM_R_SHOULDER),
                    ("l_hip", LM_L_HIP),
                    ("r_hip", LM_R_HIP),
                    ("nose", LM_NOSE),
                ]:
                    p = lm[idx]
                    row[f"{name}_x"] = p.x
                    row[f"{name}_y"] = p.y
                    row[f"{name}_vis"] = p.visibility

                l_sh, r_sh = lm[LM_L_SHOULDER], lm[LM_R_SHOULDER]
                both_visible = (
                    l_sh.visibility >= VISIBILITY_THRESHOLD
                    and r_sh.visibility >= VISIBILITY_THRESHOLD
                )
                row["both_shoulders_visible"] = both_visible
                if both_visible:
                    frames_with_both_shoulders += 1
                    row["shoulder_angle_deg"] = np.degrees(
                        np.arctan2(r_sh.y - l_sh.y, r_sh.x - l_sh.x)
                    )
                else:
                    row["shoulder_angle_deg"] = None

                draw_landmarks(frame, lm, w, h)
            else:
                for name in ["l_shoulder", "r_shoulder", "l_hip", "r_hip", "nose"]:
                    row[f"{name}_x"] = None
                    row[f"{name}_y"] = None
                    row[f"{name}_vis"] = None
                row["both_shoulders_visible"] = False
                row["shoulder_angle_deg"] = None

            writer.write(frame)
            rows.append(row)
            frame_idx += 1
    finally:
        landmarker.close()

    cap.release()
    writer.release()

    csv_path = os.path.join(out_dir, f"{clip_name}_angles.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    # Unwrap the shoulder angle across frames where it was measurable.
    known = [(r["frame"], r["shoulder_angle_deg"]) for r in rows if r["shoulder_angle_deg"] is not None]
    unwrapped_deg = None
    total_rotation_deg = 0.0
    est_chakkars = 0.0
    if len(known) >= 2:
        frames_known = [f for f, _ in known]
        angles_deg = np.array([a for _, a in known])
        unwrapped = np.degrees(np.unwrap(np.radians(angles_deg)))
        unwrapped_deg = list(zip(frames_known, unwrapped))
        total_rotation_deg = unwrapped[-1] - unwrapped[0]
        est_chakkars = abs(total_rotation_deg) / 360.0

    # Plot raw (wrapped) angle and unwrapped angle vs frame.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    if known:
        frames_known = [f for f, _ in known]
        angles_deg = [a for _, a in known]
        axes[0].plot(frames_known, angles_deg, ".", markersize=3)
        axes[0].set_ylabel("Shoulder angle (deg, wrapped)")
        axes[0].set_title(f"{clip_name}: shoulder-line orientation")
        if unwrapped_deg:
            uf, ua = zip(*unwrapped_deg)
            axes[1].plot(uf, ua, ".", markersize=3, color="darkorange")
        axes[1].set_ylabel("Unwrapped angle (deg)")
        axes[1].set_xlabel("Frame")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{clip_name}_angle_plot.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    return {
        "clip": clip_name,
        "total_frames": total_frames,
        "frames_with_pose": frames_with_pose,
        "frames_with_both_shoulders": frames_with_both_shoulders,
        "total_rotation_deg": total_rotation_deg,
        "est_chakkars": est_chakkars,
        "overlay_path": overlay_path,
        "csv_path": csv_path,
        "plot_path": plot_path,
    }


def main():
    results = []
    for filename, true_count in CLIPS.items():
        video_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(video_path):
            print(f"SKIP (not found): {video_path}")
            continue
        clip_name = os.path.splitext(filename)[0]
        out_dir = os.path.join(OUT_DIR, clip_name)
        print(f"Processing {filename}...")
        r = process_video(video_path, out_dir)
        r["true_count"] = true_count
        r["rounded_est"] = round(r["est_chakkars"])
        r["error"] = r["rounded_est"] - true_count
        results.append(r)

    print("\n=== Detection quality ===")
    print(f"{'clip':<16}{'pose%':>8}{'shoulders%':>12}")
    for r in results:
        pose_pct = 100 * r["frames_with_pose"] / r["total_frames"]
        sh_pct = 100 * r["frames_with_both_shoulders"] / r["total_frames"]
        print(f"{r['clip']:<16}{pose_pct:>7.1f}%{sh_pct:>11.1f}%")

    print("\n=== Chakkar count: shoulder-atan2 method ===")
    print(f"{'clip':<16}{'true':>6}{'raw_est':>10}{'rounded':>9}{'error':>7}")
    for r in results:
        print(
            f"{r['clip']:<16}{r['true_count']:>6}{r['est_chakkars']:>10.2f}"
            f"{r['rounded_est']:>9}{r['error']:>7}"
        )

    print("\nOutputs written under:", OUT_DIR)
    print("Inspect the *_overlay.mp4 files to manually check tracking quality,")
    print("and the *_angle_plot.png files to see whether the unwrapped angle")
    print("climbs cleanly (good sign) or is noisy/discontinuous (bad sign).")


if __name__ == "__main__":
    main()
