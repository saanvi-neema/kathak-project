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
from chakkar_scoring import score_chakkar_events  # noqa: E402
from beat_sync_check import windowed_sync_check  # noqa: E402
from report_assembly import flags_from_chakkar, flags_from_beat_sync, assemble_report  # noqa: E402
from score_aggregation import chakkar_quality_score, timing_accuracy_score, overall_score  # noqa: E402
from comparison import compare_performances  # noqa: E402

# Same BlazePose connection subset used in the pilot scripts' overlay drawing.
BODY_CONNECTIONS = [
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"), ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"), ("right_heel", "right_foot_index"),
]

# Standard 21-point MediaPipe Hands connections.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


BODY_JOINT_NAMES = sorted({name for pair in BODY_CONNECTIONS for name in pair})


def _draw_body_skeleton(canvas, row, w, h):
    for a, b in BODY_CONNECTIONS:
        ax, ay = row.get(f"body_{a}_x"), row.get(f"body_{a}_y")
        bx, by = row.get(f"body_{b}_x"), row.get(f"body_{b}_y")
        if pd.notna(ax) and pd.notna(ay) and pd.notna(bx) and pd.notna(by):
            cv2.line(canvas, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)), (109, 91, 208), 4)
    # Only the joints actually used in BODY_CONNECTIONS -- not every raw
    # BlazePose point (which would include face landmarks and clutter what's
    # supposed to read as a clean stick figure).
    for name in BODY_JOINT_NAMES:
        x, y = row.get(f"body_{name}_x"), row.get(f"body_{name}_y")
        if pd.notna(x) and pd.notna(y):
            radius = 16 if name == "nose" else 7  # draw the head bigger so it actually reads as a head
            cv2.circle(canvas, (int(x * w), int(y * h)), radius, (109, 91, 208), -1)


def _draw_hand_skeleton(canvas, row, w, h):
    for side in ["left", "right"]:
        if f"hand_{side}_0_x" not in row.index:
            continue
        pts = []
        for i in range(21):
            x, y = row.get(f"hand_{side}_{i}_x"), row.get(f"hand_{side}_{i}_y")
            pts.append((int(x * w), int(y * h)) if pd.notna(x) and pd.notna(y) else None)
        for a, b in HAND_CONNECTIONS:
            if pts[a] and pts[b]:
                cv2.line(canvas, pts[a], pts[b], (6, 119, 217), 3)
        for p in pts:
            if p:
                cv2.circle(canvas, p, 4, (6, 119, 217), -1)


DROPOUT_HOLD_FRAMES = 6  # ~0.2s at 30fps -- brief tracking loss holds last position; longer gaps stay missing


def _smooth_dropouts(df):
    """
    Landmark tracking briefly drops a joint here and there even in a mostly-
    good clip (confirmed directly: legs vanished for a few frames mid-clip
    on a fast-spin test video while everything else stayed tracked). Without
    smoothing, that reads as limbs randomly flickering in and out on the
    skeleton view. Hold the last known position for a short gap -- a real,
    sustained loss of tracking still shows as the limb disappearing, this
    only papers over brief single/few-frame blips.
    """
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
    df = df.copy()
    df[coord_cols] = df[coord_cols].ffill(limit=DROPOUT_HOLD_FRAMES)
    return df


def generate_pose_overlay(video_path, landmarks_csv, output_path):
    """
    Draws the body + hand skeleton (from the already-extracted landmarks CSV,
    not a fresh MediaPipe pass) on top of the real video frames. Writes an
    intermediate mp4v file, then re-encodes to H.264 -- mp4v (OpenCV's
    default) isn't reliably playable in a browser <video> tag, H.264 is.
    """
    df = pd.read_csv(landmarks_csv)
    df = _smooth_dropouts(df)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_path = output_path + ".raw.mp4"
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(df):
            row = df.iloc[frame_idx]
            _draw_body_skeleton(frame, row, w, h)
            _draw_hand_skeleton(frame, row, w, h)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # -vsync cfr + fixed -g (keyframe interval) forces a truly constant frame
    # rate with regular keyframes -- without this, browsers can play the
    # video fine start-to-finish once but corrupt/blank out on seek or
    # replay, a known issue with programmatically-generated H.264.
    gop = max(1, int(round(fps)))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-y", "-i", raw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-r", str(fps), "-vsync", "cfr", "-g", str(gop), "-keyint_min", str(gop),
         "-movflags", "+faststart", output_path],
        capture_output=True,
    )
    os.remove(raw_path)
    return result.returncode == 0 and os.path.exists(output_path)


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

    # Drift-during-spin (chakkar_scoring.compute_drift) needs a position
    # reference and a body-scale unit to normalize it against -- raw pixel
    # distance means nothing without knowing camera distance, but shoulder
    # width stays roughly constant regardless of that.
    out["shoulder_width"] = np.where(both_visible, np.hypot(r_x - l_x, r_y - l_y), np.nan)
    if "body_left_hip_x" in df.columns and "body_right_hip_x" in df.columns:
        l_hip_x, l_hip_y = df["body_left_hip_x"], df["body_left_hip_y"]
        r_hip_x, r_hip_y = df["body_right_hip_x"], df["body_right_hip_y"]
        com_ready = both_visible & l_hip_x.notna() & r_hip_x.notna()
        out["center_of_mass_x"] = np.where(com_ready, (l_x + r_x + l_hip_x + r_hip_x) / 4, np.nan)
        out["center_of_mass_y"] = np.where(com_ready, (l_y + r_y + l_hip_y + r_hip_y) / 4, np.nan)
    else:
        out["center_of_mass_x"] = np.nan
        out["center_of_mass_y"] = np.nan
    return out


def run_chakkar_analysis(landmarks_csv):
    """
    Segments the clip into distinct rotation bursts (chakkar_scoring.
    score_chakkar_events) and scores each independently, instead of
    unwrapping and scoring the whole clip as a single blended event -- a
    real piece can contain several separate chakkar phrases with other
    movement in between, and a single aggregate number for the whole clip
    hides that (confirmed directly: a real test clip with several distinct
    spin bursts previously scored as one falsely "clean" result with no
    flags at all). Returns None if no segment in the clip clears the
    speed/duration/rotation filters in chakkar_scoring.py.
    """
    angles_df = build_shoulder_angles(landmarks_csv)
    if angles_df is None:
        return None

    known = angles_df.dropna(subset=["shoulder_angle_deg"])
    if len(known) < 2:
        return None

    tmp_csv = landmarks_csv + "_angles_tmp.csv"
    angles_df.to_csv(tmp_csv, index=False)
    try:
        segment_results = score_chakkar_events(tmp_csv)
    finally:
        os.remove(tmp_csv)

    if not segment_results:
        return None

    events = []
    all_flags = []
    for result in segment_results:
        quality = chakkar_quality_score(result)
        flags = flags_from_chakkar(result, result["end_sec"])
        events.append({"result": result, "quality_score": quality, "flags": flags})
        all_flags.extend(flags)

    overall_quality = float(np.mean([e["quality_score"] for e in events]))
    return {"events": events, "quality_score": overall_quality, "flags": all_flags, "event_count": len(events)}


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


MIN_COMPARISON_ONSETS = 3  # too few onsets in a track to say anything meaningful about alignment


def _onset_signal(video_path, tmp_wav_path):
    """
    Extracts audio and computes what run_comparison needs: an onset-strength
    envelope (used for fastdtw alignment) and discrete onset event times
    (used for the actual action-by-action count comparison). Same
    MIN_AUDIO_RMS gate as run_timing_analysis -- a near-silent/room-noise
    track produces onset artifacts that aren't real signal.
    """
    if not extract_audio(video_path, tmp_wav_path):
        return None
    try:
        y, sr = librosa.load(tmp_wav_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        if float(np.mean(rms)) < MIN_AUDIO_RMS:
            return None
        env = librosa.onset.onset_strength(y=y, sr=sr)
        env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        return {"env": env, "env_times": env_times, "onset_times": onset_times}
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def run_comparison(teacher_video_path, student_video_path, work_dir):
    """
    Teacher-vs-student comparison (methods.md Comparison tab). Returns None
    if either track doesn't have enough audio signal to align/compare
    meaningfully, rather than faking a result.
    """
    os.makedirs(work_dir, exist_ok=True)
    teacher_wav = os.path.join(work_dir, "teacher_audio_tmp.wav")
    student_wav = os.path.join(work_dir, "student_audio_tmp.wav")

    teacher_signal = _onset_signal(teacher_video_path, teacher_wav)
    student_signal = _onset_signal(student_video_path, student_wav)
    if teacher_signal is None or student_signal is None:
        return None
    if (len(teacher_signal["onset_times"]) < MIN_COMPARISON_ONSETS
            or len(student_signal["onset_times"]) < MIN_COMPARISON_ONSETS):
        return None

    result = compare_performances(
        teacher_signal["env"], teacher_signal["env_times"], teacher_signal["onset_times"],
        student_signal["env"], student_signal["env_times"], student_signal["onset_times"],
    )
    return {
        "actions": result["actions"],
        "dense_sections": result["dense_sections"],
        "extra_teacher_sections": result["extra_teacher_sections"],
        "extra_student_sections": result["extra_student_sections"],
        "match_rate": result["match_rate"],
        "flags": result["flags"],
        "report_lines": assemble_report(result["flags"]),
    }


def analyze_video(video_path, work_dir):
    """Run the full available pipeline on one uploaded video. Returns a results dict for the frontend."""
    os.makedirs(work_dir, exist_ok=True)
    duration_sec = get_video_duration(video_path)

    landmarks_csv = extract_landmarks(video_path, output_dir=work_dir)
    features_csv = extract_features(landmarks_csv, output_dir=work_dir)

    chakkar = run_chakkar_analysis(landmarks_csv)
    timing = run_timing_analysis(video_path, features_csv, duration_sec)

    overlay_filename = "pose_overlay.mp4"
    overlay_path = os.path.join(work_dir, overlay_filename)
    overlay_ok = generate_pose_overlay(video_path, landmarks_csv, overlay_path)

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
        "overlay_video_filename": overlay_filename if overlay_ok else None,
    }
