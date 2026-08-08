"""
Phase 2A + 2B: Feature Extraction (body + hands)
Turns a Phase 1 landmarks CSV (from extract_landmarks.py) into per-frame
interpretable features:

  Body (2A): joint angles, torso tilt, head alignment, left/right symmetry,
             center of mass, velocity/acceleration/smoothness of key joints,
             foot position/trajectory (video-side of footwork, 2C)
  Hands (2B): per-finger curl angle + extended/curled flag, thumb-to-fingertip
              distances, hand orientation

This computes raw geometric measurements only -- it does NOT classify mudras
or score correctness against a reference (that needs validated ground truth
and reference mudra definitions, neither of which exist yet).

Usage:
    python extract_features.py --landmarks data/landmarks/movement_01.csv
    python extract_features.py --landmarks data/landmarks/movement_01.csv --output-dir data/features
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ---- Body joint angle definitions: (proximal, joint, distal) landmark name triples ----
BODY_ANGLE_JOINTS = {
    "l_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "r_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "l_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "r_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "l_knee": ("left_hip", "left_knee", "left_ankle"),
    "r_knee": ("right_hip", "right_knee", "right_ankle"),
    "l_hip": ("left_shoulder", "left_hip", "left_knee"),
    "r_hip": ("right_shoulder", "right_hip", "right_knee"),
}

# Landmarks tracked for velocity/acceleration/smoothness (key joints, not all 33 --
# a full-body sweep would be mostly redundant with the angle features above).
MOTION_LANDMARKS = [
    "left_wrist", "right_wrist", "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
]

# 21-point hand topology: finger name -> (mcp, pip, dip, tip) landmark indices.
HAND_FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
HAND_WRIST = 0

FPS_ASSUMED_GAP_MS = 200  # gaps larger than this aren't interpolated across


def angle_at(a, b, c):
    """Angle at vertex b (degrees) formed by points a-b-c. NaN-safe."""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba, axis=-1)
    bc_norm = np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.sum(ba * bc, axis=-1) / (ba_norm * bc_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def angle_from_vertical(vec):
    """Angle (degrees) between a 2D vector and the vertical "up" axis (0, -1)."""
    vec = np.asarray(vec, dtype=float)
    up = np.array([0.0, -1.0])
    norm = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = (vec[..., 0] * up[0] + vec[..., 1] * up[1]) / norm
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def xy(df, name):
    return df[[f"body_{name}_x", f"body_{name}_y"]].to_numpy(dtype=float)


def compute_body_angles(df, feat):
    for angle_name, (p1, p2, p3) in BODY_ANGLE_JOINTS.items():
        feat[f"angle_{angle_name}"] = angle_at(xy(df, p1), xy(df, p2), xy(df, p3))


def compute_posture(df, feat):
    l_sh, r_sh = xy(df, "left_shoulder"), xy(df, "right_shoulder")
    l_hip, r_hip = xy(df, "left_hip"), xy(df, "right_hip")
    nose = xy(df, "nose")

    mid_shoulder = (l_sh + r_sh) / 2
    mid_hip = (l_hip + r_hip) / 2

    feat["torso_tilt_deg"] = angle_from_vertical(mid_shoulder - mid_hip)
    feat["head_tilt_deg"] = angle_from_vertical(nose - mid_shoulder)

    # Approximate center of mass as the centroid of shoulders + hips -- a coarse
    # proxy (not a true segmental-mass-weighted COM), adequate for a relative/
    # comparative signal within one performance.
    feat["center_of_mass_x"] = (l_sh[:, 0] + r_sh[:, 0] + l_hip[:, 0] + r_hip[:, 0]) / 4
    feat["center_of_mass_y"] = (l_sh[:, 1] + r_sh[:, 1] + l_hip[:, 1] + r_hip[:, 1]) / 4

    feat["shoulder_level_diff"] = l_sh[:, 1] - r_sh[:, 1]
    feat["hip_level_diff"] = l_hip[:, 1] - r_hip[:, 1]


def compute_symmetry(feat):
    pairs = [("elbow", "l_elbow", "r_elbow"), ("shoulder", "l_shoulder", "r_shoulder"),
              ("knee", "l_knee", "r_knee"), ("hip", "l_hip", "r_hip")]
    diffs = []
    for label, l_key, r_key in pairs:
        d = (feat[f"angle_{l_key}"] - feat[f"angle_{r_key}"]).abs()
        feat[f"symmetry_{label}_diff"] = d
        diffs.append(d)
    feat["symmetry_mean_diff"] = pd.concat(diffs, axis=1).mean(axis=1)


def compute_foot_trajectory(df, feat):
    l_ankle, r_ankle = xy(df, "left_ankle"), xy(df, "right_ankle")
    feat["foot_separation"] = np.linalg.norm(l_ankle - r_ankle, axis=-1)
    feat["foot_height_diff"] = l_ankle[:, 1] - r_ankle[:, 1]


def compute_motion(df, feat, t_sec):
    """Velocity, acceleration, and a jerk-based smoothness signal for key landmarks."""
    dt = np.gradient(t_sec)
    dt[dt == 0] = np.nan

    for name in MOTION_LANDMARKS:
        pos = xy(df, name)
        valid = ~np.isnan(pos).any(axis=1)

        # Smooth with a Savitzky-Golay filter where enough consecutive valid
        # samples exist; leave NaN where tracking was lost.
        smoothed = np.full_like(pos, np.nan)
        win = min(9, len(pos) - (1 - len(pos) % 2))  # must be odd and <= len
        if win >= 5 and valid.sum() >= win:
            for axis in range(2):
                series = pd.Series(pos[:, axis]).interpolate(limit=5)
                if series.notna().sum() >= win:
                    smoothed[:, axis] = savgol_filter(series.bfill().ffill(),
                                                       window_length=win, polyorder=2)
        else:
            smoothed = pos

        vel = np.gradient(smoothed, axis=0) / dt[:, None]
        speed = np.linalg.norm(vel, axis=-1)
        accel = np.gradient(vel, axis=0) / dt[:, None]
        accel_mag = np.linalg.norm(accel, axis=-1)
        jerk = np.gradient(accel, axis=0) / dt[:, None]
        jerk_mag = np.linalg.norm(jerk, axis=-1)

        speed[~valid] = np.nan
        accel_mag[~valid] = np.nan
        jerk_mag[~valid] = np.nan

        feat[f"{name}_speed"] = speed
        feat[f"{name}_accel"] = accel_mag
        feat[f"{name}_jerk"] = jerk_mag  # lower = smoother


def hand_xy(df, side, idx):
    return df[[f"hand_{side}_{idx}_x", f"hand_{side}_{idx}_y"]].to_numpy(dtype=float)


def compute_hand_features(df, feat):
    for side in ["left", "right"]:
        cols_present = f"hand_{side}_0_x" in df.columns
        if not cols_present:
            continue

        wrist = hand_xy(df, side, HAND_WRIST)
        for finger, (mcp, pip, dip, tip) in HAND_FINGERS.items():
            mcp_xy = hand_xy(df, side, mcp)
            pip_xy = hand_xy(df, side, pip)
            dip_xy = hand_xy(df, side, dip)
            tip_xy = hand_xy(df, side, tip)

            # Curl = how far the fingertip has folded back toward the wrist,
            # measured as the angle at the PIP joint (mcp-pip-dip); ~180 deg =
            # straight/extended, smaller = curled.
            pip_angle = angle_at(mcp_xy, pip_xy, dip_xy)
            feat[f"hand_{side}_{finger}_curl_angle"] = pip_angle
            # `pip_angle > 160` on a NaN (untracked landmark, e.g. motion
            # blur/occlusion during a fast gesture) evaluates to plain False,
            # not NaN -- silently reporting "curled" for a frame with no real
            # data instead of leaving it missing. np.where keeps genuinely
            # untracked frames as NaN so a window average (see
            # app/pipeline.py's run_mudra_analysis) correctly skips them
            # instead of having them vote "curled" by default.
            feat[f"hand_{side}_{finger}_extended"] = np.where(np.isnan(pip_angle), np.nan, pip_angle > 160)

            feat[f"hand_{side}_{finger}_tip_to_wrist"] = np.linalg.norm(tip_xy - wrist, axis=-1)

        # Hand-scale reference (wrist to middle-MCP -- "palm length") so
        # distance-based features below are comparable across videos shot at
        # different distances from the camera. Real bug found and fixed: a
        # video where the hand appears smaller/larger in frame than the
        # reference photos used absolute normalized-coordinate distances,
        # so a fixed touch/spread threshold silently stopped meaning the
        # same thing -- confirmed directly when a frame that MUST be pataka
        # (guaranteed by recording order) failed pataka's own thumb-touch
        # check under the old un-normalized distances.
        middle_mcp_xy = hand_xy(df, side, HAND_FINGERS["middle"][0])
        hand_scale = np.linalg.norm(middle_mcp_xy - wrist, axis=-1)
        hand_scale = np.where(hand_scale > 1e-6, hand_scale, np.nan)  # guard div-by-zero on a degenerate frame

        # Thumb-to-other-fingertip distances -- the signal most mudra
        # definitions actually hinge on (e.g. Pataka: thumb touches index base;
        # Kartarimukh: thumb+ring touch while index+pinky spread). Normalized
        # by hand_scale so a fixed threshold means "close relative to this
        # hand's own size", not a fixed number of pixels/frame-fraction.
        thumb_tip = hand_xy(df, side, HAND_FINGERS["thumb"][3])
        for finger in ["index", "middle", "ring", "pinky"]:
            tip = hand_xy(df, side, HAND_FINGERS[finger][3])
            feat[f"hand_{side}_thumb_to_{finger}_tip"] = np.linalg.norm(thumb_tip - tip, axis=-1) / hand_scale

        # Adjacent-fingertip spread (index-middle, middle-ring, ring-pinky) --
        # distinguishes "fingers together" (Pataka) from "fingers spread"
        # (Kartarimukh, Alapadma). Also hand-scale normalized.
        adjacent = [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]
        for f1, f2 in adjacent:
            tip1 = hand_xy(df, side, HAND_FINGERS[f1][3])
            tip2 = hand_xy(df, side, HAND_FINGERS[f2][3])
            feat[f"hand_{side}_{f1}_{f2}_spread"] = np.linalg.norm(tip1 - tip2, axis=-1) / hand_scale

        # Hand orientation: angle of wrist -> middle-MCP vector from vertical,
        # a simple 2D proxy for how the hand is rotated in frame.
        middle_mcp = hand_xy(df, side, HAND_FINGERS["middle"][0])
        feat[f"hand_{side}_orientation_deg"] = angle_from_vertical(middle_mcp - wrist)


def extract_features(landmarks_csv: str, output_dir: str = "data/features") -> str:
    os.makedirs(output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(landmarks_csv))[0]
    output_path = os.path.join(output_dir, f"{clip_name}_features.csv")

    df = pd.read_csv(landmarks_csv)
    feat = pd.DataFrame({"frame": df["frame"], "timestamp_ms": df["timestamp_ms"]})
    t_sec = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0

    has_body = "body_nose_x" in df.columns
    if has_body:
        compute_body_angles(df, feat)
        compute_posture(df, feat)
        compute_symmetry(feat)
        compute_foot_trajectory(df, feat)
        compute_motion(df, feat, t_sec)
    else:
        print(f"  (no body landmarks in {landmarks_csv}, skipping 2A)")

    has_hands = any(c.startswith("hand_") for c in df.columns)
    if has_hands:
        compute_hand_features(df, feat)
    else:
        print(f"  (no hand landmarks in {landmarks_csv}, skipping 2B)")

    feat.to_csv(output_path, index=False)
    print(f"Saved {len(feat)} rows, {len(feat.columns)} columns to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract body + hand features from a Phase 1 landmarks CSV")
    parser.add_argument("--landmarks", required=True, help="Path to a landmarks CSV from extract_landmarks.py")
    parser.add_argument("--output-dir", default="data/features", help="Output directory for feature CSVs")
    args = parser.parse_args()

    extract_features(args.landmarks, args.output_dir)


if __name__ == "__main__":
    main()
