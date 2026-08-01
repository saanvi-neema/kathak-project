"""
Builds a labeled mudra training dataset from raw per-mudra video clips.

Input convention: data/mudra_training/raw/<mudra_name>.mov (or .mp4/.avi/
.mkv/.m4v) -- one file per mudra, each containing several hold/relax/reform
reps (see methods.md for the exact recording instructions this matches).
Since each file only contains ONE mudra, there's no ground-truth-timestamp
matching problem like the earlier stuck attempt at this -- the filename IS
the label. The only per-frame work needed is separating the "held" (mudra
actually formed) frames from the "relaxing between reps" transition frames,
so the dataset isn't contaminated with frames that don't actually look like
the mudra.

Held-frame detection: a sliding window over total hand-landmark displacement
(all 21 points, frame to frame) -- a sustained LOW-motion stretch is a held
pose, a high-motion stretch is a transition. Same windowed-threshold pattern
already validated for chakkar burst segmentation (chakkar_scoring.
segment_rotation_bursts), applied to a different signal here.

STILL_THRESHOLD and the duration constants below are reasoned defaults, NOT
calibrated against real mudra footage -- none exists yet. The script prints
a per-clip held-rep count specifically so that's easy to sanity-check once
real clips exist, and to retune quickly if the threshold is off.

Usage:
    python build_mudra_training_data.py
    python build_mudra_training_data.py --raw-dir data/mudra_training/raw --output data/mudra_training/dataset.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from extract_landmarks import extract_landmarks  # noqa: E402

STILL_WINDOW_SEC = 0.3
STILL_STEP_SEC = 0.1
STILL_THRESHOLD = 0.01       # mean per-point frame-to-frame displacement (normalized coords) below this reads as "held" -- not calibrated against real footage yet
MIN_HOLD_DURATION_SEC = 0.5  # a held stretch must last at least this long to count as one rep, not a brief pause mid-motion
HOLD_EDGE_TRIM_SEC = 0.15    # trimmed off each end of a detected hold, in case the edges still have residual settling motion

VIDEO_EXTENSIONS = (".mov", ".mp4", ".avi", ".mkv", ".m4v")


def hand_points(df, side):
    """(n_frames, 21, 2) array of a hand's landmark positions, or None if that hand isn't in this CSV at all."""
    coords = []
    for i in range(21):
        xcol, ycol = f"hand_{side}_{i}_x", f"hand_{side}_{i}_y"
        if xcol not in df.columns:
            return None
        coords.append(df[[xcol, ycol]].to_numpy(dtype=float))
    return np.stack(coords, axis=1)


def hand_motion_magnitude(df, side):
    """
    Per-frame hand-shape motion: mean frame-to-frame displacement across all
    21 landmark points. Low = hand shape is stable (holding a mudra); high =
    the hand is actively moving/reforming. First frame is NaN (no prior
    frame to compare against).
    """
    pts = hand_points(df, side)
    if pts is None:
        return None
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=-1)  # (n_frames-1, 21)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A row where this hand wasn't tracked at all is legitimately
        # all-NaN (e.g. a clip where only the other hand performs the
        # mudra) -- numpy's "Mean of empty slice" warning on that is
        # expected noise, not a bug to chase.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        magnitude = np.nanmean(diffs, axis=1)
    return np.concatenate([[np.nan], magnitude])


def find_held_windows(t, motion, window_sec=STILL_WINDOW_SEC, step_sec=STILL_STEP_SEC,
                       threshold=STILL_THRESHOLD, min_duration_sec=MIN_HOLD_DURATION_SEC):
    """
    Sliding-window version of chakkar_scoring.segment_rotation_bursts'
    pattern, inverted: finds sustained LOW-motion stretches instead of
    high-motion ones. Returns a list of (start_sec, end_sec) tuples.
    """
    t = np.asarray(t)
    motion = np.asarray(motion)
    if len(t) < 2:
        return []

    centers, still = [], []
    start = t[0]
    while start + window_sec <= t[-1]:
        end = start + window_sec
        mask = (t >= start) & (t <= end)
        if mask.sum() >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                window_motion = np.nanmean(motion[mask])
            still.append((not np.isnan(window_motion)) and window_motion < threshold)
            centers.append((start + end) / 2)
        start += step_sec

    if not centers:
        return []
    centers = np.array(centers)
    still = np.array(still)

    raw_segments = []
    in_seg = False
    seg_start = None
    for i, is_still in enumerate(still):
        if is_still and not in_seg:
            seg_start = centers[i]
            in_seg = True
        if not is_still and in_seg:
            raw_segments.append((seg_start, centers[i - 1]))
            in_seg = False
    if in_seg:
        raw_segments.append((seg_start, centers[-1]))

    return [(s, e) for s, e in raw_segments if (e - s) >= min_duration_sec]


def build_dataset_for_mudra(video_path, mudra_name, landmarks_dir):
    """
    Extracts landmarks for one raw clip and returns a DataFrame of just its
    held-pose frames, labeled. Only one hand is used per clip, even if both
    are tracked -- recording instructions only ask for one hand to perform
    the mudra, so an idle hand resting in frame would stay nearly still for
    the whole clip and, if not excluded, would get mislabeled as holding
    the mudra too. The hand with more total motion across the clip is
    assumed to be the one actually doing the reps.
    """
    landmarks_csv = extract_landmarks(video_path, output_dir=landmarks_dir)
    df = pd.read_csv(landmarks_csv)
    t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0

    motions = {}
    for side in ["left", "right"]:
        motion = hand_motion_magnitude(df, side)
        if motion is not None:
            motions[side] = motion

    if not motions:
        return None

    active_side = max(motions, key=lambda s: np.nansum(motions[s]))
    motion = motions[active_side]

    held_windows = find_held_windows(t, motion)
    print(f"  {mudra_name} ({active_side} hand): {len(held_windows)} held rep(s) found")
    if not held_windows:
        return None

    rows = []
    for start, end in held_windows:
        trimmed_start, trimmed_end = start + HOLD_EDGE_TRIM_SEC, end - HOLD_EDGE_TRIM_SEC
        if trimmed_end <= trimmed_start:
            trimmed_start, trimmed_end = start, end  # too short to trim without losing everything
        mask = (t >= trimmed_start) & (t <= trimmed_end)
        held_frames = df[mask].copy()
        held_frames["mudra"] = mudra_name
        held_frames["hand_side"] = active_side
        rows.append(held_frames)

    return pd.concat(rows, ignore_index=True)


def build_all(raw_dir, landmarks_dir, output_csv):
    if not os.path.isdir(raw_dir):
        print(f"Raw clip directory not found: {raw_dir}")
        return

    files = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(VIDEO_EXTENSIONS))
    if not files:
        print(f"No video files found in {raw_dir}")
        return

    all_rows = []
    for fname in files:
        mudra_name = os.path.splitext(fname)[0].lower()
        print(f"Processing {fname} -> mudra '{mudra_name}'")
        result = build_dataset_for_mudra(os.path.join(raw_dir, fname), mudra_name, landmarks_dir)
        if result is not None:
            all_rows.append(result)
        else:
            print(f"  WARNING: no held frames found for {mudra_name} -- check STILL_THRESHOLD or the clip itself")

    if not all_rows:
        print("No training rows produced from any clip.")
        return

    dataset = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dataset.to_csv(output_csv, index=False)
    print(f"\nSaved {len(dataset)} labeled frames across {dataset['mudra'].nunique()} mudras to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Build a labeled mudra training dataset from raw per-mudra clips")
    parser.add_argument("--raw-dir", default="data/mudra_training/raw")
    parser.add_argument("--landmarks-dir", default="data/mudra_training/landmarks")
    parser.add_argument("--output", default="data/mudra_training/dataset.csv")
    args = parser.parse_args()
    build_all(args.raw_dir, args.landmarks_dir, args.output)


if __name__ == "__main__":
    main()
