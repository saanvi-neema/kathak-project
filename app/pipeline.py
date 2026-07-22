"""
Dashboard backend glue: runs the analysis pipeline on an uploaded video and
returns structured results for the frontend.

Chakkar analysis and timing/beat-sync analysis run fully automatically on
any video that has body pose and/or audio. Mudra analysis is NOT run
automatically here -- mudra_reference.py can only check a NAMED mudra
against its rule, it can't identify which mudra is happening at a given
moment (no classifier exists, no training data for one yet). The Mudra tab
reports this honestly rather than faking a result.
"""

import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from extract_landmarks import extract_landmarks  # noqa: E402
from extract_features import extract_features  # noqa: E402
from chakkar_scoring import score_chakkar, describe as describe_chakkar  # noqa: E402
from beat_sync_check import windowed_sync_check  # noqa: E402
from report_assembly import flags_from_chakkar, flags_from_beat_sync, assemble_report  # noqa: E402
from score_aggregation import chakkar_quality_score, timing_accuracy_score, overall_score  # noqa: E402

MIN_ROTATION_DEG_FOR_CHAKKAR = 180  # below this, don't bother scoring "chakkar quality"


def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (total / fps) if fps else 0.0


def extract_audio(video_path, out_wav):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", out_wav],
        capture_output=True,
    )
    return result.returncode == 0 and os.path.exists(out_wav)


def build_shoulder_angles(landmarks_csv):
    """
    chakkar_scoring.score_chakkar() expects a 'shoulder_angle_deg' column
    (the format chakkar_pilot.py produces). extract_landmarks.py's general
    output has separate left/right shoulder x/y instead -- compute the same
    atan2 angle from that here so any uploaded video can be scored, not just
    the original pilot clips.
    """
    df = pd.read_csv(landmarks_csv)
    required = ["body_left_shoulder_x", "body_left_shoulder_y", "body_right_shoulder_x", "body_right_shoulder_y"]
    if not all(c in df.columns for c in required):
        return None

    out = pd.DataFrame({"frame": df["frame"], "timestamp_ms": df["timestamp_ms"]})
    l_x, l_y = df["body_left_shoulder_x"], df["body_left_shoulder_y"]
    r_x, r_y = df["body_right_shoulder_x"], df["body_right_shoulder_y"]
    both_visible = l_x.notna() & r_x.notna()
    out["shoulder_angle_deg"] = np.where(
        both_visible, np.degrees(np.arctan2(r_y - l_y, r_x - l_x)), np.nan
    )
    return out


def run_chakkar_analysis(landmarks_csv):
    angles_df = build_shoulder_angles(landmarks_csv)
    if angles_df is None:
        return None

    known = angles_df.dropna(subset=["shoulder_angle_deg"])
    if len(known) < 2:
        return None

    unwrapped = np.degrees(np.unwrap(np.radians(known["shoulder_angle_deg"].to_numpy())))
    total_rotation = abs(unwrapped[-1] - unwrapped[0])
    if total_rotation < MIN_ROTATION_DEG_FOR_CHAKKAR:
        return None  # not enough rotation in this clip to call it a chakkar

    tmp_csv = landmarks_csv + "_angles_tmp.csv"
    angles_df.to_csv(tmp_csv, index=False)
    try:
        result = score_chakkar(tmp_csv)
    finally:
        os.remove(tmp_csv)

    end_time_sec = angles_df["timestamp_ms"].max() / 1000.0
    quality = chakkar_quality_score(result)
    flags = flags_from_chakkar(result, end_time_sec)
    return {"result": result, "quality_score": quality, "flags": flags, "end_time_sec": end_time_sec}


MIN_AUDIO_RMS = 0.02  # below this, treat it as room noise/silence, not real music
MIN_DURATION_FOR_WINDOWED_CHECK = 8.0  # matches windowed_sync_check's default window_sec


def run_timing_analysis(video_path, features_csv, duration_sec):
    wav_path = video_path + "_audio_tmp.wav"
    if not extract_audio(video_path, wav_path):
        return None

    try:
        y, sr = librosa.load(wav_path, sr=None)

        # A tempo estimate on near-silent audio is a known artifact (a fixed
        # fallback value shows up regardless of content -- confirmed earlier
        # this session on room-noise-only clips), not a real detected tempo.
        # Don't report it at all rather than show a misleading number.
        rms = librosa.feature.rms(y=y)[0]
        if float(np.mean(rms)) < MIN_AUDIO_RMS:
            return None

        if duration_sec < MIN_DURATION_FOR_WINDOWED_CHECK:
            return None  # too short for even one sync-check window to run

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 4:
            return None  # not enough beat signal to say anything meaningful

        feat = pd.read_csv(features_csv)
        if "right_wrist_speed" not in feat.columns:
            return None
        t_sec = feat["timestamp_ms"].to_numpy(dtype=float) / 1000.0
        speed = feat["right_wrist_speed"].to_numpy()

        from scipy.signal import find_peaks
        valid_speed = np.nan_to_num(speed, nan=0)
        threshold = np.nanpercentile(speed, 75) if np.any(~np.isnan(speed)) else 0
        peaks, _ = find_peaks(valid_speed, height=threshold, distance=5)
        peak_times = t_sec[peaks]

        windowed = windowed_sync_check(peak_times, beat_times, clip_duration=duration_sec,
                                        window_sec=8.0, step_sec=2.0, min_events=5)
        accuracy = timing_accuracy_score(windowed, clip_duration_sec=duration_sec) if windowed else None
        flags = flags_from_beat_sync(windowed, category="timing") if windowed else []
        return {"tempo_bpm": float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo),
                "windowed_results": windowed, "accuracy_score": accuracy, "flags": flags}
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def analyze_video(video_path, work_dir):
    """Run the full available pipeline on one uploaded video. Returns a results dict for the frontend."""
    os.makedirs(work_dir, exist_ok=True)
    duration_sec = get_video_duration(video_path)

    landmarks_csv = extract_landmarks(video_path, output_dir=work_dir)
    features_csv = extract_features(landmarks_csv, output_dir=work_dir)

    chakkar = run_chakkar_analysis(landmarks_csv)
    timing = run_timing_analysis(video_path, features_csv, duration_sec)

    all_flags = []
    if chakkar:
        all_flags.extend(chakkar["flags"])
    if timing:
        all_flags.extend(timing["flags"])
    report_lines = assemble_report(all_flags)

    overall = overall_score(
        mudra_score=None,  # not run automatically -- see module docstring
        chakkar_score=chakkar["quality_score"] if chakkar else None,
        timing_score=timing["accuracy_score"] if timing else None,
    )

    return {
        "duration_sec": duration_sec,
        "chakkar": chakkar,
        "timing": timing,
        "mudra": None,  # honestly not run -- no automatic mudra identification exists yet
        "overall_score": overall,
        "report_lines": report_lines,
    }
