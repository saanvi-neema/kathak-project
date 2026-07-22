"""
Regression tests for chakkar_scoring.py against the 3 ground-truth clips
(1/3/10 spins) -- this is the one piece of the whole project that's been
most rigorously validated by hand. These tests exist so a future change
can't silently break it without anything catching it.
"""

import os

import pytest

from chakkar_scoring import score_chakkar

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "landmarks", "chakkar_pilot",
)

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
