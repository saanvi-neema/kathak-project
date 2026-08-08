"""
Dashboard backend glue: runs the analysis pipeline on an uploaded video and
returns structured results for the frontend.

Chakkar analysis and timing/beat-sync analysis run fully automatically on
any video that has body pose and/or audio. Mudra identification runs
automatically too IF a trained classifier exists at MUDRA_MODEL_PATH (see
mudra_classifier.py/build_mudra_training_data.py) -- otherwise it honestly
reports None rather than faking a result, since mudra_reference.py alone can
only check a NAMED mudra against its rule, not identify which mudra is
happening at a given moment.
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
from extract_features import extract_features, compute_hand_features  # noqa: E402
from chakkar_scoring import score_chakkar_events  # noqa: E402
from beat_sync_check import windowed_sync_check  # noqa: E402
from report_assembly import flags_from_chakkar, flags_from_beat_sync, flags_from_mudra_checks, assemble_report  # noqa: E402
from score_aggregation import chakkar_quality_score, timing_accuracy_score, mudra_rule_agreement_score, mudra_identification_accuracy_score, count_checked_constraints, overall_score  # noqa: E402
from comparison import compare_performances, find_dense_stretches  # noqa: E402

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


def _beat_grid_from_audio(y, sr, duration_sec):
    """
    Pure beat-grid logic (RMS gate, duration gate, tempo/beat detection),
    split out of _extract_beat_grid so a live capture session's in-memory
    rolling audio buffer can reuse it directly, without a redundant
    wav-file round trip. Returns (tempo_bpm, beat_times) or None if there's
    not enough real audio signal to trust a beat grid at all -- same gates
    _extract_beat_grid always used (a near-silent/room-noise track produces
    a fixed fallback tempo artifact, not a real detected one).
    """
    rms = librosa.feature.rms(y=y)[0]
    if float(np.mean(rms)) < MIN_AUDIO_RMS:
        return None
    if duration_sec < MIN_DURATION_FOR_WINDOWED_CHECK:
        return None
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 4:
        return None
    tempo_bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    return tempo_bpm, beat_times


def _extract_beat_grid(video_path, duration_sec):
    """
    Shared beat-grid extraction for run_timing_analysis and
    run_taal_analysis, so both don't decode the same audio twice. Extracts
    audio from the given video file, then delegates the actual detection
    logic to _beat_grid_from_audio.
    """
    wav_path = video_path + "_beatgrid_tmp.wav"
    if not extract_audio(video_path, wav_path):
        return None
    try:
        y, sr = librosa.load(wav_path, sr=None)
        return _beat_grid_from_audio(y, sr, duration_sec)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


ENDING_OFFSET_FLAG_THRESHOLD_SEC = 0.5  # a reasoned deadzone (same spirit as TAAL_OFFSET_FLAG_THRESHOLD), not calibrated against real labeled endings yet -- none exists in this project


def _flags_from_ending(duration_sec, beat_times):
    """
    Compares the clip's actual end time to the last beat in the detected
    beat grid -- flags a performance that finished noticeably before or
    after the music's last beat (methods.md step 6's "you ended 1 second
    before the ending beat"). Pure logic, no audio I/O -- same split as
    _taal_flags_from_chakkar, directly testable with a synthetic beat grid.
    """
    from beat_sync_check import last_beat_offset_sec
    from report_assembly import Flag

    offset = last_beat_offset_sec(duration_sec, beat_times)
    if offset is None or abs(offset) < ENDING_OFFSET_FLAG_THRESHOLD_SEC:
        return []
    direction = "before" if offset < 0 else "after"
    return [Flag(
        duration_sec, "timing",
        f"you ended about {abs(offset):.2f} seconds {direction} the ending beat."
    )]


def run_timing_analysis(video_path, features_csv, duration_sec, beat_grid=None):
    """
    beat_grid: pass a precomputed (tempo_bpm, beat_times) tuple (e.g. from
    _beat_grid_from_audio) to skip re-extracting audio from video_path -- a
    live capture session already has its rolling audio buffer in memory.
    Falls back to the normal video_path-based extraction if not supplied.
    """
    if beat_grid is None:
        beat_grid = _extract_beat_grid(video_path, duration_sec)
    if beat_grid is None:
        return None
    tempo_bpm, beat_times = beat_grid

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
    flags.extend(_flags_from_ending(duration_sec, beat_times))
    return {"tempo_bpm": tempo_bpm, "windowed_results": windowed, "accuracy_score": accuracy, "flags": flags}


TAAL_OFFSET_FLAG_THRESHOLD = 0.15  # beats; a reasoned deadzone (same spirit as chakkar's count_gap_threshold), not calibrated against real taal-labeled audio yet -- none exists in this project (see taal_reference.py)


def _taal_flags_from_chakkar(chakkar_events, beat_times, taal_name, sam_time):
    """
    Pure logic, no audio I/O -- given already-extracted beat_times and a
    list of chakkar_scoring events, computes each chakkar's signed distance
    from the nearest sam (taal_reference.beats_from_sam) and flags any that
    land far enough off to be a real landing error, not measurement noise.
    Kept separate from run_taal_analysis() so this is directly testable
    with synthetic data, same pattern as comparison.py's core functions.
    """
    from taal_reference import beats_from_sam
    from report_assembly import Flag

    events, flags = [], []
    for event in chakkar_events:
        end_sec = event["result"]["end_sec"]
        offset = beats_from_sam(end_sec, beat_times, taal_name, sam_time)
        if offset is None:
            continue
        events.append({
            "start_sec": event["result"]["start_sec"], "end_sec": end_sec,
            "beats_from_sam": float(offset),
        })
        if abs(offset) >= TAAL_OFFSET_FLAG_THRESHOLD:
            direction = "before" if offset < 0 else "after"
            flags.append(Flag(
                end_sec, "taal",
                f"your chakkar ended about {abs(offset):.2f} beats {direction} sam, not landing clean on the cycle."
            ))
    return events, flags


def run_taal_analysis(video_path, duration_sec, chakkar, taal_name, sam_time, beat_grid=None):
    """
    Checks chakkar landings against sam using the real taal cycle structure
    (taal_reference.py). Requires taal_name + sam_time to be supplied --
    neither is auto-detected (see taal_reference.py's module docstring for
    why). Returns None if any of chakkar/taal_name/sam_time are missing, or
    if there isn't enough audio signal for a beat grid.

    beat_grid: pass a precomputed (tempo_bpm, beat_times) tuple to skip
    re-extracting audio from video_path -- see run_timing_analysis's
    docstring for why (live capture sessions already hold this in memory).
    """
    if not taal_name or sam_time is None or not chakkar or not chakkar.get("events"):
        return None

    from taal_reference import TAAL_DEFINITIONS
    if taal_name not in TAAL_DEFINITIONS:
        return None

    if beat_grid is None:
        beat_grid = _extract_beat_grid(video_path, duration_sec)
    if beat_grid is None:
        return None
    _, beat_times = beat_grid

    events, flags = _taal_flags_from_chakkar(chakkar["events"], beat_times, taal_name, sam_time)
    if not events:
        return None
    return {"taal": taal_name, "sam_time": sam_time, "events": events, "flags": flags}


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


def run_tatkaar_analysis(video_path, work_dir):
    """
    Single-video tatkaar (rhythmic footwork) detection -- pipeline step 7,
    previously unbuilt (see methods.md). Audio-based, not pose-based: a
    tatkaar phrase produces a sustained run of closely-spaced percussive
    onsets (foot/ghungroo strikes), which is exactly what comparison.py's
    find_dense_stretches() was already built to detect for teacher-vs-
    student comparison -- reused here unmodified, just run on one video's
    own onsets instead of comparing two tracks.

    UNVALIDATED against real tatkaar footage -- find_dense_stretches()'s
    thresholds (DENSE_GAP_THRESHOLD_SEC, MIN_DENSE_STRETCH_ONSETS) are
    reasoned defaults, never checked against a real recording (none exists
    in this project yet -- see methods.md). Also inherits comparison.py's
    honest limitation: an onset can't be told apart from a clap or any
    other percussive movement sound, only counted.

    Returns None if there isn't enough audio signal to detect onsets at all
    (_onset_signal's RMS gate), or no dense stretch is found -- same
    "honestly report nothing" pattern as the rest of this pipeline, not a
    forced zero-strike result.
    """
    tmp_wav_path = os.path.join(work_dir, "tatkaar_audio_tmp.wav")
    signal = _onset_signal(video_path, tmp_wav_path)
    if signal is None:
        return None

    stretches = find_dense_stretches(signal["onset_times"])
    if not stretches:
        return None

    events = []
    for s in stretches:
        duration = s["end"] - s["start"]
        events.append({
            "start_sec": s["start"],
            "end_sec": s["end"],
            "strike_count": s["count"],
            "strikes_per_sec": (s["count"] / duration) if duration > 0 else None,
        })
    return {"events": events}


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


MUDRA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mudra_training", "model.joblib")
# A prediction below this isn't reported. Originally calibrated against 23
# real, individually-verified hold-windows from mudra_01.mov (see
# methods.md) -- correct predictions averaged confidence 0.44, wrong ones
# averaged 0.29, a real but not clean separation. 0.30 was chosen to trade
# some precision for much better recall (see methods.md for the full
# reasoning).
#
# model.joblib was retrained (predict_mudra() added a required
# feature_means bundle key the old on-disk model predated -- the old model
# raised on every prediction until this retrain). A spot-check against
# mudra_01.mov's real footage after retraining shows the same shape of
# result (24 detected hold-windows, confidence mean ~0.46, predicted labels
# in the same relative order as the previously hand-verified correct
# answers), but the rigorous per-window accuracy number above has NOT been
# re-verified against the retrained model -- that required a human
# checking each hold against the real recording order, which hasn't been
# redone yet. Treat the 23-data-point numbers above as pre-retrain history,
# not a current calibration guarantee, until that re-check happens.
MUDRA_MIN_CONFIDENCE = 0.30


def run_mudra_analysis(landmarks_csv, model_path=MUDRA_MODEL_PATH, expected_sequence=None, model_bundle=None):
    """
    Identifies which mudra (if any) is being held during each stable hand
    pose in the video (mudra_classifier.py), then checks that identified
    mudra's hand shape against its geometric reference rule
    (mudra_reference.py) -- combining "what mudra is this" with "is it
    formed correctly", which was the actual target behavior described early
    in this project, not just identification alone.

    That rule check is NOT identification accuracy -- it only tests whether
    the observed hand shape agrees with whatever mudra the classifier
    guessed, never whether the guess itself was right (a real circularity,
    caught in an external review -- see methods.md step 4 and
    score_aggregation.mudra_rule_agreement_score's docstring). To get an
    actual identification-accuracy signal, pass `expected_sequence`: an
    ordered list of the mudra names the dancer actually intended to perform
    (e.g. from a known practice sequence, same idea as mudra_01.mov's
    verified recording order). Each detected hold, in time order, is
    compared positionally against that list -- event i vs expected_sequence[i]
    -- and gets `expected_mudra`/`matches_expected` fields. If the counts
    don't line up (extra or missed detections), the trailing events just
    get no expected label (`None`) rather than a forced, likely-wrong pairing.

    Returns None if no trained model exists yet at model_path -- the honest
    default until real mudra training footage is recorded and
    mudra_classifier.py is run (see methods.md); the dashboard already
    reports that plainly instead of faking a result.

    model_bundle: pass an already-loaded bundle (joblib.load(model_path)'s
    return value) to skip loading it from disk. The real model file is
    174MB -- a live capture session calls this every ~1-2 seconds and needs
    to load it once per session, not once per call.
    """
    from build_mudra_training_data import hand_motion_magnitude, find_held_windows
    from mudra_classifier import predict_mudra
    from mudra_reference import check_mudra_variants, MUDRA_RULES

    bundle = model_bundle
    if bundle is None:
        if not os.path.exists(model_path):
            return None
        import joblib
        bundle = joblib.load(model_path)

    df = pd.read_csv(landmarks_csv)
    t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0

    events = []
    for side in ["left", "right"]:
        motion = hand_motion_magnitude(df, side)
        if motion is None:
            continue
        for start, end in find_held_windows(t, motion):
            mask = (t >= start) & (t <= end)
            segment_df = df[mask]
            if segment_df.empty:
                continue

            feat = pd.DataFrame(index=segment_df.index)
            compute_hand_features(segment_df, feat)
            side_cols = [c for c in feat.columns if c.startswith(f"hand_{side}_")]
            if not side_cols:
                continue

            mean_row = feat[side_cols].mean().to_dict()
            for col in list(mean_row):
                if col.endswith("_extended"):
                    # mean() over a held window's True/False values gives a
                    # fraction, not a bool -- check_mudra()'s bool() cast
                    # would treat ANY nonzero fraction as "extended", so this
                    # needs an explicit majority-vote threshold first.
                    mean_row[col] = mean_row[col] > 0.5

            stripped = {c[len(f"hand_{side}_"):]: v for c, v in mean_row.items()}
            label, confidence = predict_mudra(bundle, stripped)
            if label is None or confidence < MUDRA_MIN_CONFIDENCE:
                continue

            mismatches, constraints_checked = None, 0
            try:
                variant_name, mismatches = check_mudra_variants(mean_row, label, side=side)
                constraints_checked = count_checked_constraints(MUDRA_RULES[variant_name])
            except ValueError:
                pass  # classifier predicted a label with no matching reference rule -- report identification only

            events.append({
                "start_sec": float(start), "end_sec": float(end),
                "hand_side": side, "mudra": label, "confidence": float(confidence),
                "mismatches": mismatches, "constraints_checked": constraints_checked,
            })

    events.sort(key=lambda e: e["start_sec"])
    _attach_expected_mudras(events, expected_sequence)
    return events if events else None


def _attach_expected_mudras(events, expected_sequence):
    """
    Pure positional-matching logic, pulled out of run_mudra_analysis so it's
    testable without needing to synthesize real hand-motion data to produce
    multiple held windows. `events` (already time-sorted) get an
    `expected_mudra`/`matches_expected` field each, matching event i against
    expected_sequence[i]. Extra events past the end of expected_sequence are
    left unmatched (None/None) rather than force-paired with nothing.
    """
    expected_sequence = expected_sequence or []
    for i, e in enumerate(events):
        if i < len(expected_sequence):
            e["expected_mudra"] = expected_sequence[i]
            e["matches_expected"] = (e["mudra"] == expected_sequence[i])
        else:
            e["expected_mudra"] = None
            e["matches_expected"] = None


RASA_SAMPLE_INTERVAL_SEC = 1.0  # fixed sampling clock -- no "expression is stable now" detector exists yet, unlike mudra's held-window detection of hand poses


def run_rasa_analysis(landmarks_csv):
    """
    Classifies which navarasa the face most closely matches in fixed
    1-second windows across the clip (rasa_reference.classify_rasa),
    averaging blendshapes within each window to smooth single-frame noise.
    Samples on a fixed clock rather than detecting held-expression windows
    the way run_mudra_analysis detects held hand poses, since no equivalent
    "expression is stable now" detector exists for faces yet.

    UNVALIDATED against real deliberate-abhinaya footage -- rasa_reference.py's
    thresholds are reasoned defaults, not calibrated against real labeled
    data (see methods.md). Reported plainly as a rough read, the same
    honesty gap mudra classification had before mudra_01.mov existed to
    test against. No accuracy score is computed here (unlike mudra/timing)
    since there's nothing real yet to calibrate one against.

    Returns None if the landmarks CSV has no face blendshape columns at all
    (e.g. the face was never in frame -- see extract_landmarks.py).
    """
    from rasa_reference import classify_rasa, RASA_DEFINITIONS

    df = pd.read_csv(landmarks_csv)
    face_cols = [c for c in df.columns if c.startswith("face_")]
    if not face_cols:
        return None

    t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    if len(t) == 0:
        return None
    duration = float(t[-1])

    events = []
    window_start = 0.0
    while window_start <= duration:
        window_end = window_start + RASA_SAMPLE_INTERVAL_SEC
        mask = (t >= window_start) & (t < window_end)
        window = df.loc[mask, face_cols].dropna(how="all")
        if not window.empty:
            mean_row = {k: v for k, v in window.mean().to_dict().items() if pd.notna(v)}
            best, mismatches, scores = classify_rasa(mean_row)
            events.append({
                "start_sec": window_start,
                "end_sec": min(window_end, duration),
                "rasa": best,
                "display_name": RASA_DEFINITIONS[best]["display_name"],
                "confidence_label": RASA_DEFINITIONS[best]["confidence"],
                "mismatch_count": len(mismatches),
            })
        window_start += RASA_SAMPLE_INTERVAL_SEC

    return events if events else None


def analyze_video(video_path, work_dir, taal_name=None, sam_time=None, expected_mudra_sequence=None):
    """
    Run the full available pipeline on one uploaded video. Returns a
    results dict for the frontend. taal_name/sam_time are optional --
    supply both to get chakkar-vs-sam checks (see run_taal_analysis);
    neither is auto-detected, so without them taal analysis is just
    honestly skipped, same as every other optional signal in this pipeline.

    expected_mudra_sequence is also optional -- an ordered list of mudra
    names the dancer intended to perform, if known. Without it, mudra
    scoring is limited to rule-agreement (see run_mudra_analysis's
    docstring for why that's not the same as accuracy); with it, a real
    identification-accuracy score becomes possible and takes priority in
    overall_score.
    """
    os.makedirs(work_dir, exist_ok=True)
    duration_sec = get_video_duration(video_path)

    landmarks_csv = extract_landmarks(video_path, output_dir=work_dir)
    features_csv = extract_features(landmarks_csv, output_dir=work_dir)

    chakkar = run_chakkar_analysis(landmarks_csv)
    # Extracted once and shared with run_taal_analysis below -- both need the
    # same audio decode + tempo/beat detection over this video, and
    # run_timing_analysis always needs it regardless of whether taal analysis
    # ends up running at all, so there's no cost added by extracting it here
    # up front instead of inside run_timing_analysis.
    beat_grid = _extract_beat_grid(video_path, duration_sec)
    timing = run_timing_analysis(video_path, features_csv, duration_sec, beat_grid=beat_grid)
    mudra_events = run_mudra_analysis(landmarks_csv, expected_sequence=expected_mudra_sequence)
    taal = run_taal_analysis(video_path, duration_sec, chakkar, taal_name, sam_time, beat_grid=beat_grid)
    rasa_events = run_rasa_analysis(landmarks_csv)
    tatkaar = run_tatkaar_analysis(video_path, work_dir)

    overlay_filename = "pose_overlay.mp4"
    overlay_path = os.path.join(work_dir, overlay_filename)
    overlay_ok = generate_pose_overlay(video_path, landmarks_csv, overlay_path)

    mudra_flags = []
    mudra_check_results = []
    mudra_id_matches = []
    if mudra_events:
        for e in mudra_events:
            if e["mismatches"] is not None:
                mudra_flags.extend(flags_from_mudra_checks([(e["end_sec"], e["mudra"], e["mismatches"])]))
                mudra_check_results.append((e["mismatches"], e["constraints_checked"]))
            if e["matches_expected"] is not None:
                mudra_id_matches.append(e["matches_expected"])
    mudra_rule_score = mudra_rule_agreement_score(mudra_check_results) if mudra_check_results else None
    mudra_id_score = mudra_identification_accuracy_score(mudra_id_matches) if mudra_id_matches else None
    # Prefer real identification accuracy (checked against an actual expected
    # label) over rule-agreement (checked only against the classifier's own
    # guess) wherever both exist -- see run_mudra_analysis's docstring.
    mudra_score_for_overall = mudra_id_score if mudra_id_score is not None else mudra_rule_score

    all_flags = []
    if chakkar:
        all_flags.extend(chakkar["flags"])
    if timing:
        all_flags.extend(timing["flags"])
    if taal:
        all_flags.extend(taal["flags"])
    all_flags.extend(mudra_flags)
    report_lines = assemble_report(all_flags)

    overall = overall_score(
        mudra_score=mudra_score_for_overall,
        chakkar_score=chakkar["quality_score"] if chakkar else None,
        timing_score=timing["accuracy_score"] if timing else None,
    )

    mudra = {
        "events": mudra_events,
        "rule_agreement_score": mudra_rule_score,
        "identification_accuracy_score": mudra_id_score,
        "flags": mudra_flags,
    } if mudra_events else None

    return {
        "duration_sec": duration_sec,
        "chakkar": chakkar,
        "timing": timing,
        "taal": taal,  # None unless both taal_name and sam_time were supplied -- see run_taal_analysis
        "mudra": mudra,  # None until a trained classifier exists at MUDRA_MODEL_PATH -- see run_mudra_analysis
        "rasa": rasa_events,  # UNVALIDATED against real footage -- see run_rasa_analysis. Not part of overall_score or report_lines.
        "tatkaar": tatkaar,  # UNVALIDATED against real footage -- see run_tatkaar_analysis. Not part of overall_score or report_lines.
        "overall_score": overall,
        "report_lines": report_lines,
        "overlay_video_filename": overlay_filename if overlay_ok else None,
    }
