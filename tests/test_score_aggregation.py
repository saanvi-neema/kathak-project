"""
Tests for score_aggregation.py -- in particular the "deadzone" fix (a
performance with no flag-worthy deviations must score exactly 100, not
lose fractional points for measurement noise too small to ever be
mentioned in the report). This was a real bug caught once already; this
test exists so it can't come back silently.
"""

import pytest

from score_aggregation import chakkar_quality_score, overall_score, timing_accuracy_score


def make_chakkar_result(count_gap=0.0, orientation_gap_deg=0.0, controlled_stop=True):
    return {
        "count_gap": count_gap,
        "orientation_gap_deg": orientation_gap_deg,
        "controlled_stop": controlled_stop,
    }


def test_deviations_below_flag_threshold_score_100():
    """Mirrors the real chakkar_02 case: tiny count/orientation gaps that
    never cross report_assembly's flagging thresholds must not cost points."""
    result = make_chakkar_result(count_gap=0.0016, orientation_gap_deg=1.1, controlled_stop=True)
    assert chakkar_quality_score(result) == 100.0


def test_deviation_just_below_threshold_still_scores_100():
    result = make_chakkar_result(count_gap=0.049, orientation_gap_deg=14.9, controlled_stop=True)
    assert chakkar_quality_score(result) == 100.0


def test_deviation_past_threshold_costs_points():
    result = make_chakkar_result(count_gap=0.2, orientation_gap_deg=0.0, controlled_stop=True)
    score = chakkar_quality_score(result)
    assert score < 100.0, "a real, flag-worthy deviation should cost something"
    assert score > 0.0


def test_uncontrolled_stop_costs_points_even_with_perfect_count():
    clean = make_chakkar_result(count_gap=0.0, orientation_gap_deg=0.0, controlled_stop=True)
    abrupt = make_chakkar_result(count_gap=0.0, orientation_gap_deg=0.0, controlled_stop=False)
    assert chakkar_quality_score(abrupt) < chakkar_quality_score(clean)


def test_overall_score_averages_only_available_categories():
    assert overall_score(chakkar_score=100.0) == 100.0
    assert overall_score(chakkar_score=100.0, timing_score=50.0) == 75.0
    assert overall_score(mudra_score=90.0, chakkar_score=100.0, timing_score=80.0) == pytest.approx(90.0)


def test_overall_score_none_when_nothing_available():
    assert overall_score() is None


def test_timing_accuracy_full_marks_when_nothing_unsynced():
    windowed = [
        {"window_start": 0, "window_end": 8, "synchronized": True},
        {"window_start": 2, "window_end": 10, "synchronized": True},
    ]
    assert timing_accuracy_score(windowed, clip_duration_sec=20.0) == 100.0


def test_timing_accuracy_reflects_unsynced_duration():
    # mirrors the real tatkaar reference clip finding: ~16s unsynced out of ~42s
    windowed = [
        {"window_start": 20, "window_end": 28, "synchronized": False},
        {"window_start": 22, "window_end": 30, "synchronized": False},
        {"window_start": 24, "window_end": 32, "synchronized": False},
        {"window_start": 26, "window_end": 34, "synchronized": False},
        {"window_start": 28, "window_end": 36, "synchronized": False},
        {"window_start": 0, "window_end": 8, "synchronized": True},
    ]
    score = timing_accuracy_score(windowed, clip_duration_sec=41.9)
    # merged unsynced span is 20-36s = 16s out of 41.9s
    expected = 100 * (1 - 16 / 41.9)
    assert abs(score - expected) < 1.0
