"""
Processes one live-capture chunk against a LiveSession: decodes its frames,
extends the session's rolling landmarks/audio buffers, and re-runs the
existing (unmodified) analysis functions against everything accumulated so
far. Returns a snapshot shaped like pipeline.analyze_video()'s return value
so the frontend can reuse the same rendering it already has for uploads.

See app/live_session.py for the session state this operates on, and the
live-capture plan for the overall design -- in particular why events get a
"finalize margin" before being reported (chakkar_scoring.py's segment
detection and build_mudra_training_data.py's held-window detection both
close out whatever's still in progress at the end of whatever data they're
given -- re-running them every tick on a growing buffer means something
still actively happening right now would otherwise be reported as
"finished" and then "un-finish and re-finish longer" next tick).
"""
import logging
import os
import sys
import time

import cv2
import librosa
import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from extract_landmarks import extract_frame_landmarks  # noqa: E402
from extract_features import extract_features  # noqa: E402
from chakkar_scoring import SEGMENT_MERGE_GAP_SEC, SEGMENT_SCORE_PAD_SEC  # noqa: E402
from build_mudra_training_data import STILL_WINDOW_SEC  # noqa: E402
from report_assembly import assemble_report, flags_from_mudra_checks  # noqa: E402
from score_aggregation import (  # noqa: E402
    mudra_rule_agreement_score, mudra_identification_accuracy_score, overall_score,
)

from pipeline import (  # noqa: E402
    run_chakkar_analysis, run_mudra_analysis, run_rasa_analysis,
    run_timing_analysis, run_taal_analysis, extract_audio, _beat_grid_from_audio,
    RASA_SAMPLE_INTERVAL_SEC,
)
from live_session import TIMING_RECOMPUTE_INTERVAL_SEC  # noqa: E402

# Margin before a segment/hold is safe to report as finished, not still
# growing -- same constants those checks already tune their own edge
# behavior with, not new numbers invented for live mode.
CHAKKAR_FINALIZE_MARGIN_SEC = SEGMENT_MERGE_GAP_SEC + SEGMENT_SCORE_PAD_SEC
MUDRA_FINALIZE_MARGIN_SEC = STILL_WINDOW_SEC

DEFAULT_CHUNK_FPS_FALLBACK = 30.0  # used only if a chunk's client-reported duration is missing/invalid

logger = logging.getLogger(__name__)


def _decode_chunk_frames(chunk_path, expected_duration_sec):
    """
    Reads every frame from one chunk video file. cv2's own reported fps and
    frame-count metadata are unreliable for these short recorder-cycled
    chunks (confirmed directly against real recorded phone chunks before
    building this: cv2 reported a nonsense fps and a garbage frame count on
    every single chunk, while a plain read-to-EOF loop worked every time) --
    effective fps for this chunk is derived from frames actually read
    divided by the client-reported capture duration instead.
    """
    cap = cv2.VideoCapture(chunk_path)
    frames = []
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    cap.release()

    if expected_duration_sec and expected_duration_sec > 0 and frames:
        effective_fps = len(frames) / expected_duration_sec
    else:
        effective_fps = DEFAULT_CHUNK_FPS_FALLBACK
    return frames, effective_fps


def _next_chunk_timestamp_ms(last_ts_ms, effective_fps):
    """
    A live session's effective fps can differ from chunk to chunk (each
    chunk is decoded independently -- see _decode_chunk_frames), unlike
    extract_landmarks.py's next_video_timestamp_ms, which assumes one
    constant fps across an entire file and computes an absolute
    frame_idx/fps position from it. Reusing that formula here with a
    frame_idx counter that keeps incrementing across chunks of *different*
    fps would jump or compress time at every chunk boundary. This instead
    advances by a per-frame delta from the last timestamp, which stays
    correct regardless of fps changing between chunks.
    """
    if last_ts_ms < 0:
        return 0.0
    return last_ts_ms + (1000.0 / effective_fps)


def _extend_landmarks(session, frames, effective_fps):
    session.effective_fps = effective_fps
    for frame in frames:
        ts_ms = _next_chunk_timestamp_ms(session.last_ts, effective_fps)
        ts_ms_int = int(round(ts_ms))
        if ts_ms_int <= session.last_ts:
            ts_ms_int = int(session.last_ts) + 1
        session.last_ts = ts_ms_int

        row = extract_frame_landmarks(
            session.pose_landmarker, session.hand_landmarker, session.face_landmarker,
            frame, session.frame_idx, ts_ms_int,
        )
        session.landmarks_rows.append(row)
        session.frame_idx += 1

    session.duration_sec = session.last_ts / 1000.0 if session.last_ts >= 0 else 0.0


def _extend_audio(session, chunk_path):
    wav_path = chunk_path + "_audio_tmp.wav"
    try:
        if not extract_audio(chunk_path, wav_path):
            return
        y, sr = librosa.load(wav_path, sr=None)
        if sr != session.sr:
            # extract_audio() always resamples to session.sr (22050 Hz) via
            # ffmpeg's -ar flag -- this should never happen, but don't
            # silently concatenate mismatched sample rates if it somehow does.
            return
        session.audio_samples = np.concatenate([session.audio_samples, y.astype(np.float32)])
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def _write_landmarks_csv(session):
    pd.DataFrame(session.landmarks_rows).to_csv(session.landmarks_csv_path, index=False)


def _merge_chakkar_events(session, chakkar_result):
    if not chakkar_result:
        return
    already = len(session.chakkar_events)
    for event in chakkar_result["events"][already:]:
        end_sec = event["result"]["end_sec"]
        if end_sec + CHAKKAR_FINALIZE_MARGIN_SEC <= session.duration_sec:
            session.chakkar_events.append(event)
        else:
            break  # events are time-ordered -- once one isn't safe yet, none after it are either


def _rebuild_chakkar_summary(session):
    """Same packaging run_chakkar_analysis itself does, built from only the finalized subset."""
    if not session.chakkar_events:
        return None
    all_flags = []
    for e in session.chakkar_events:
        all_flags.extend(e["flags"])
    overall_quality = float(np.mean([e["quality_score"] for e in session.chakkar_events]))
    return {
        "events": session.chakkar_events, "quality_score": overall_quality,
        "flags": all_flags, "event_count": len(session.chakkar_events),
    }


def _merge_rasa_events(session, rasa_result):
    """
    Keyed by start_sec, not position: run_rasa_analysis skips a window
    entirely if no face was detected anywhere in it, which would silently
    shift list positions across ticks (e.g. face lost after a couple
    seconds -- see the live-capture plan's manual verification notes) if
    events were matched up by index instead. A window's own end_sec is
    clipped to the current buffer duration while it's still accumulating
    (see run_rasa_analysis) and only reaches its full width once the
    buffer has genuinely passed it -- that's the actual finalize signal,
    not "is this the last item in the list" (which breaks the same way if
    face detection permanently stops mid-session -- the last real reading
    would otherwise never finalize).
    """
    if not rasa_result:
        return
    for event in rasa_result:
        start_sec = event["start_sec"]
        if start_sec in session.rasa_events:
            continue
        if event["end_sec"] - start_sec + 1e-6 >= RASA_SAMPLE_INTERVAL_SEC:
            session.rasa_events[start_sec] = event


def _merge_mudra_events(session, mudra_result):
    if not mudra_result:
        return
    for event in mudra_result:
        if event["end_sec"] + MUDRA_FINALIZE_MARGIN_SEC <= session.duration_sec:
            key = (event["hand_side"], event["start_sec"])
            session.mudra_events[key] = event


def _assemble_mudra_summary(session):
    events = sorted(session.mudra_events.values(), key=lambda e: e["start_sec"])
    if not events:
        return None

    mudra_flags, check_results, id_matches = [], [], []
    for e in events:
        if e["mismatches"] is not None:
            mudra_flags.extend(flags_from_mudra_checks([(e["end_sec"], e["mudra"], e["mismatches"])]))
            check_results.append((e["mismatches"], e["constraints_checked"]))
        if e["matches_expected"] is not None:
            id_matches.append(e["matches_expected"])

    return {
        "events": events,
        "rule_agreement_score": mudra_rule_agreement_score(check_results) if check_results else None,
        "identification_accuracy_score": mudra_identification_accuracy_score(id_matches) if id_matches else None,
        "flags": mudra_flags,
    }


def _snapshot(session, chakkar, mudra, rasa):
    all_flags = []
    if chakkar:
        all_flags.extend(chakkar["flags"])
    if session.timing:
        all_flags.extend(session.timing["flags"])
    if session.taal:
        all_flags.extend(session.taal["flags"])
    if mudra:
        all_flags.extend(mudra["flags"])

    mudra_score_for_overall = None
    if mudra:
        mudra_score_for_overall = (
            mudra["identification_accuracy_score"]
            if mudra["identification_accuracy_score"] is not None
            else mudra["rule_agreement_score"]
        )

    overall = overall_score(
        mudra_score=mudra_score_for_overall,
        chakkar_score=chakkar["quality_score"] if chakkar else None,
        timing_score=session.timing["accuracy_score"] if session.timing else None,
    )

    return {
        "duration_sec": session.duration_sec,
        "chakkar": chakkar,
        "timing": session.timing,
        "taal": session.taal,
        "mudra": mudra,
        "rasa": rasa,
        "overall_score": overall,
        "report_lines": assemble_report(all_flags),
        "overlay_video_filename": None,
        "original_video_filename": None,
    }


def current_snapshot(session):
    """
    Rebuilds the current results snapshot from a session's already-finalized
    events, without processing a new chunk -- used by /live/stop to return a
    final state.
    """
    chakkar = _rebuild_chakkar_summary(session)
    mudra = _assemble_mudra_summary(session)
    rasa = sorted(session.rasa_events.values(), key=lambda e: e["start_sec"]) if session.rasa_events else None
    return _snapshot(session, chakkar=chakkar, mudra=mudra, rasa=rasa)


def process_live_chunk(session, chunk_bytes, ext, chunk_duration_sec):
    chunk_path = os.path.join(session.work_dir, f"chunk_tmp.{ext}")
    with open(chunk_path, "wb") as f:
        f.write(chunk_bytes)

    try:
        frames, effective_fps = _decode_chunk_frames(chunk_path, chunk_duration_sec)
        if frames:
            _extend_landmarks(session, frames, effective_fps)
        _extend_audio(session, chunk_path)
    finally:
        if os.path.exists(chunk_path):
            os.remove(chunk_path)

    if not session.landmarks_rows:
        return _snapshot(session, chakkar=None, mudra=None, rasa=None)

    _write_landmarks_csv(session)
    csv_path = session.landmarks_csv_path

    chakkar_raw = run_chakkar_analysis(csv_path)
    _merge_chakkar_events(session, chakkar_raw)
    chakkar = _rebuild_chakkar_summary(session)

    # A live tick every ~1-2s over a session that can run for minutes means
    # one bad prediction shouldn't take down the whole snapshot the way an
    # uncaught exception would for a single batch /analyze call -- report
    # mudra as unavailable for this tick and keep going, same "honestly
    # report nothing instead of crashing" spirit run_mudra_analysis itself
    # already uses for a missing model file.
    try:
        mudra_raw = run_mudra_analysis(
            csv_path, expected_sequence=session.expected_mudra_sequence, model_bundle=session.model_bundle,
        )
    except Exception:
        logger.exception("run_mudra_analysis failed mid-live-session %s", session.session_id)
        mudra_raw = None
    _merge_mudra_events(session, mudra_raw)
    mudra = _assemble_mudra_summary(session)

    rasa_raw = run_rasa_analysis(csv_path)
    _merge_rasa_events(session, rasa_raw)
    rasa = sorted(session.rasa_events.values(), key=lambda e: e["start_sec"]) if session.rasa_events else None

    now = time.time()
    if now - session.last_timing_recompute_wall >= TIMING_RECOMPUTE_INTERVAL_SEC:
        session.last_timing_recompute_wall = now
        beat_grid = _beat_grid_from_audio(session.audio_samples, session.sr, session.duration_sec)
        session.cached_beat_grid = beat_grid
        if beat_grid is not None:
            features_csv = extract_features(csv_path, output_dir=session.work_dir)
            session.timing = run_timing_analysis(features_csv, session.duration_sec, beat_grid)
            session.taal = run_taal_analysis(
                session.duration_sec, chakkar, session.taal_name, session.sam_time, beat_grid,
            )
        else:
            # Not enough real audio signal accumulated yet -- same "honestly
            # report nothing" behavior the batch pipeline has for short/quiet
            # clips, not a live-mode-specific gap.
            session.timing = None
            session.taal = None

    return _snapshot(session, chakkar=chakkar, mudra=mudra, rasa=rasa)
