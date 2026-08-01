"""
Tests for build_mudra_training_data.py's pure signal-processing logic:
detecting "held" (stable, mudra actually formed) frames vs. "transition"
(relaxing/reforming between reps) frames from raw hand-landmark motion.
Only the numeric logic is tested here with synthetic landmark data -- the
video-processing/extract_landmarks parts need real footage, which doesn't
exist yet.
"""

import numpy as np
import pandas as pd
import pytest

from build_mudra_training_data import hand_motion_magnitude, find_held_windows


def make_hand_df(t, moving_mask=None, move_amount=0.1, jitter=0.0005, seed=0):
    """A synthetic 21-point hand landmark DataFrame: still by default, with
    an optional injected motion stretch (moving_mask) simulating a reformed/
    relaxed hand between reps."""
    rng = np.random.default_rng(seed)
    n = len(t)
    cols = {}
    for i in range(21):
        x = np.full(n, 0.5)
        y = np.full(n, 0.5)
        if moving_mask is not None:
            x[moving_mask] += np.linspace(0, move_amount, moving_mask.sum())
        x = x + rng.normal(0, jitter, n)
        cols[f"hand_left_{i}_x"] = x
        cols[f"hand_left_{i}_y"] = y
    df = pd.DataFrame(cols)
    df["timestamp_ms"] = t * 1000
    return df


def test_hand_motion_magnitude_is_low_when_still_high_when_moving():
    t = np.arange(0, 3, 1 / 30)
    moving = (t >= 1.0) & (t < 1.5)
    df = make_hand_df(t, moving_mask=moving)

    motion = hand_motion_magnitude(df, "left")
    still_motion = np.nanmean(motion[(t > 0.1) & (t < 0.9)])
    moving_motion = np.nanmean(motion[(t >= 1.0) & (t < 1.5)])
    assert moving_motion > still_motion * 5


def test_hand_motion_magnitude_returns_none_when_hand_missing():
    df = pd.DataFrame({"timestamp_ms": [0, 33, 66]})
    assert hand_motion_magnitude(df, "left") is None


def test_find_held_windows_splits_around_a_transition():
    """A held pose, a reform/relax stretch, then another held pose should
    read as two separate held windows, not one continuous one."""
    t = np.arange(0, 4.5, 1 / 30)
    moving = (t >= 2.0) & (t < 2.5)
    df = make_hand_df(t, moving_mask=moving)
    motion = hand_motion_magnitude(df, "left")

    windows = find_held_windows(t, motion)
    assert len(windows) == 2
    first_end = windows[0][1]
    second_start = windows[1][0]
    assert first_end < 2.0 + 0.3  # shouldn't extend well past where motion actually started
    assert second_start > 2.2     # shouldn't start well before motion actually ended


def test_find_held_windows_one_continuous_hold_is_one_window():
    t = np.arange(0, 2, 1 / 30)
    df = make_hand_df(t)  # no motion injected at all
    motion = hand_motion_magnitude(df, "left")

    windows = find_held_windows(t, motion)
    assert len(windows) == 1


def test_find_held_windows_brief_pause_does_not_count_as_a_hold():
    """A pause too short to be a real rep (below MIN_HOLD_DURATION_SEC)
    shouldn't be reported as its own held window."""
    t = np.arange(0, 3, 1 / 30)
    moving = (t < 1.4) | (t > 1.5)  # only ~0.1s of stillness in the middle
    df = make_hand_df(t, moving_mask=moving, move_amount=0.05)
    motion = hand_motion_magnitude(df, "left")

    windows = find_held_windows(t, motion, min_duration_sec=0.5)
    assert all((e - s) >= 0.5 for s, e in windows)
