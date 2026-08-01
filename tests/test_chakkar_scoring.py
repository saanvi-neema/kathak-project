"""
Regression tests for chakkar_scoring.py against the 3 ground-truth clips
(1/3/10 spins) -- this is the one piece of the whole project that's been
most rigorously validated by hand. These tests exist so a future change
can't silently break it without anything catching it.
"""

import os

import numpy as np
import pandas as pd
import pytest

from chakkar_scoring import score_chakkar, score_chakkar_events, segment_rotation_bursts, compute_drift

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "landmarks", "chakkar_pilot")
MOVEMENT_01_LANDMARKS_CSV = os.path.join(PROJECT_ROOT, "data", "landmarks", "movement_01.csv")

# (clip name, true spin count, count tolerance)
GROUND_TRUTH = [
    ("chakkar_01", 1.0),
    ("chakkar_02", 3.0),
    ("chakkar_03", 10.0),
]


@pytest.mark.parametrize("clip_name,true_count", GROUND_TRUTH)
def test_rotation_count_matches_ground_truth(clip_name, true_count):
    angles_csv = os.path.join(DATA_DIR, clip_name, f"{clip_name}_angles.csv")
    result = score_chakkar(angles_csv)

    assert result["rounded_count"] == true_count, (
        f"{clip_name}: rounded count {result['rounded_count']} != ground truth {true_count}"
    )
    # raw count should be close to the true count, not just rounding luckily --
    # this is what "no drift over more rotations" actually means as a check.
    assert abs(result["raw_count"] - true_count) < 0.1, (
        f"{clip_name}: raw count {result['raw_count']:.3f} strayed too far from {true_count}"
    )


@pytest.mark.parametrize("clip_name,_", GROUND_TRUTH)
def test_all_ground_truth_clips_end_facing_front(clip_name, _):
    """All 3 pilot clips were confirmed (against the actual video frames) to
    end facing the same direction they started -- a real chakkar should."""
    angles_csv = os.path.join(DATA_DIR, clip_name, f"{clip_name}_angles.csv")
    result = score_chakkar(angles_csv)
    assert abs(result["orientation_gap_deg"]) < 15, (
        f"{clip_name}: orientation gap {result['orientation_gap_deg']:.1f} deg exceeds tolerance"
    )


@pytest.mark.parametrize("clip_name,_", GROUND_TRUTH)
def test_all_ground_truth_clips_have_controlled_stop(clip_name, _):
    """All 3 pilot clips are clean, deliberate performances -- none should
    read as an abrupt/interrupted stop."""
    angles_csv = os.path.join(DATA_DIR, clip_name, f"{clip_name}_angles.csv")
    result = score_chakkar(angles_csv)
    assert bool(result["controlled_stop"]), f"{clip_name}: expected a controlled stop"


# --- Multi-event segmentation ---------------------------------------------
#
# The rest of this file covers segment_rotation_bursts / score_chakkar_events
# -- the fix for a real bug found during development: scoring a whole clip
# as one blended event hid multiple separate chakkar phrases in a real test
# clip (movement_01.mov, a Birju Maharaj piece excerpt) behind a single
# falsely-clean result. These tests pin that fix in place: the 3 ground
# truth clips (each one continuous spin) must still segment as exactly one
# event, and movement_01.mov must be recognized as containing several
# distinct rotation bursts, not one.

@pytest.mark.parametrize("clip_name,true_count", GROUND_TRUTH)
def test_ground_truth_clips_segment_as_a_single_event(clip_name, true_count):
    """A clip that's one continuous spin shouldn't get artificially split
    into multiple segments -- segmentation must agree with score_chakkar()
    on these already-validated clips, not just produce a different answer."""
    angles_csv = os.path.join(DATA_DIR, clip_name, f"{clip_name}_angles.csv")
    events = score_chakkar_events(angles_csv)
    assert len(events) == 1, f"{clip_name}: expected exactly 1 segment, got {len(events)}"
    assert events[0]["rounded_count"] == true_count
    assert abs(events[0]["raw_count"] - true_count) < 0.1


def _shoulder_angles_from_landmarks(landmarks_csv):
    """
    Minimal standalone version of app/pipeline.py's build_shoulder_angles --
    duplicated here (not imported) so this test file stays independent of
    the app layer's heavier dependencies (cv2, imageio_ffmpeg).
    """
    df = pd.read_csv(landmarks_csv)
    l_x, l_y = df["body_left_shoulder_x"], df["body_left_shoulder_y"]
    r_x, r_y = df["body_right_shoulder_x"], df["body_right_shoulder_y"]
    both_visible = l_x.notna() & r_x.notna()
    angle = np.where(both_visible, np.degrees(np.arctan2(r_y - l_y, r_x - l_x)), np.nan)
    return pd.DataFrame({"frame": df["frame"], "timestamp_ms": df["timestamp_ms"], "shoulder_angle_deg": angle})


@pytest.mark.skipif(not os.path.exists(MOVEMENT_01_LANDMARKS_CSV), reason="movement_01.mov landmarks not extracted locally")
def test_mixed_clip_finds_multiple_rotation_bursts(tmp_path):
    """
    The actual regression case: movement_01.mov previously scored as one
    "clean" chakkar (raw_count=2.01, no flags) despite containing several
    separate rotation bursts. Segmentation must find more than one distinct
    candidate burst here -- the whole point of this fix.
    """
    angles_df = _shoulder_angles_from_landmarks(MOVEMENT_01_LANDMARKS_CSV)
    known = angles_df.dropna(subset=["shoulder_angle_deg"])
    t = known["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    unwrapped = np.degrees(np.unwrap(np.radians(known["shoulder_angle_deg"].to_numpy())))

    segments = segment_rotation_bursts(t, unwrapped)
    assert len(segments) > 1, "expected multiple distinct rotation bursts in a mixed real clip"

    angles_csv = tmp_path / "movement_01_angles.csv"
    angles_df.to_csv(angles_csv, index=False)
    events = score_chakkar_events(str(angles_csv))
    # At least one segment should clear the full chakkar bar; each scored
    # event must carry its own start/end, not the whole clip's.
    assert len(events) >= 1
    for event in events:
        assert event["end_sec"] - event["start_sec"] < 10, "a scored event should be a short burst, not most of the clip"


def test_single_frame_glitch_is_not_treated_as_a_rotation_burst():
    """
    A one-frame tracking glitch (e.g. a momentary left/right shoulder swap)
    can look like a huge instantaneous angle jump. Confirmed on real data
    that this shows up as exactly one 0.1s window spiking over threshold
    with normal values immediately before and after -- segmentation must not
    treat that as a real sustained rotation.
    """
    dt = 1 / 30
    n = 90  # 3 seconds at 30fps
    times = np.arange(n) * dt
    angles = np.zeros(n)  # perfectly still...
    glitch_idx = 45
    angles[glitch_idx] = 170.0  # ...except one single-frame spike

    segments = segment_rotation_bursts(times, angles)
    assert segments == []


def test_sustained_rotation_is_detected_as_a_burst():
    """Sanity check in the other direction: a real sustained fast rotation must be found."""
    dt = 1 / 30
    n = 90
    times = np.arange(n) * dt
    # constant 400 deg/s for the whole 3 seconds -- well past both the speed
    # and minimum-duration thresholds.
    angles = 400.0 * times

    segments = segment_rotation_bursts(times, angles)
    assert len(segments) == 1
    assert segments[0]["end"] - segments[0]["start"] > 1.0


# --- Drift-during-spin ------------------------------------------------------
#
# compute_drift() is deliberately NOT a pass/fail check (a traveling chakkar
# can be intentional) -- these tests just pin the two numbers it reports:
# net displacement normalized by shoulder width, and path straightness
# (net distance / total distance covered).

def _make_drift_df(times, com_x, com_y, shoulder_width=0.1):
    return pd.DataFrame({
        "timestamp_ms": times * 1000.0,
        "center_of_mass_x": com_x,
        "center_of_mass_y": com_y,
        "shoulder_width": shoulder_width,
    })


def test_straight_line_travel_has_high_straightness():
    """A dancer moving steadily in one direction (a deliberate traveling
    chakkar) should measure close to straightness=1.0 -- net and total
    distance covered are nearly the same."""
    times = np.linspace(0, 2, 40)
    com_x = np.linspace(0.3, 0.6, 40)  # steady rightward move
    com_y = np.full(40, 0.5)
    df = _make_drift_df(times, com_x, com_y, shoulder_width=0.1)

    drift = compute_drift(df, start_sec=0.0, end_sec=2.0, pad_sec=0.0)
    assert drift is not None
    assert drift["path_straightness"] > 0.95
    # net displacement 0.3 in normalized coords / 0.1 shoulder width = 3.0 shoulder widths
    assert drift["drift_shoulder_widths"] == pytest.approx(3.0, rel=0.05)


def test_wobbly_back_and_forth_has_low_straightness_even_with_small_net_drift():
    """Lots of back-and-forth movement that mostly cancels out should read
    as a low straightness ratio, even though net displacement is small --
    this is the "wandered a lot but didn't really go anywhere" case."""
    times = np.linspace(0, 2, 40)
    # oscillates back and forth with a tiny net drift
    com_x = 0.5 + 0.05 * np.sin(np.linspace(0, 6 * np.pi, 40)) + np.linspace(0, 0.01, 40)
    com_y = np.full(40, 0.5)
    df = _make_drift_df(times, com_x, com_y, shoulder_width=0.1)

    drift = compute_drift(df, start_sec=0.0, end_sec=2.0, pad_sec=0.0)
    assert drift is not None
    assert drift["path_straightness"] < 0.3
    assert drift["drift_shoulder_widths"] < 0.5  # net drift is small despite all the wobbling


def test_no_movement_has_straightness_one_and_zero_drift():
    times = np.linspace(0, 2, 10)
    com_x = np.full(10, 0.5)
    com_y = np.full(10, 0.5)
    df = _make_drift_df(times, com_x, com_y, shoulder_width=0.1)

    drift = compute_drift(df, start_sec=0.0, end_sec=2.0, pad_sec=0.0)
    assert drift is not None
    assert drift["drift_shoulder_widths"] == pytest.approx(0.0)
    assert drift["path_straightness"] == 1.0


def test_missing_center_of_mass_columns_returns_none():
    df = pd.DataFrame({"timestamp_ms": [0, 1000, 2000]})
    assert compute_drift(df, 0.0, 2.0) is None


def test_too_few_frames_returns_none():
    df = _make_drift_df(np.array([0.0]), np.array([0.5]), np.array([0.5]))
    assert compute_drift(df, 0.0, 0.0) is None
