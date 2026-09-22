"""
Tests for score_aggregation.py -- in particular the "deadzone" fix (a
performance with no flag-worthy deviations must score exactly 100, not
lose fractional points for measurement noise too small to ever be
mentioned in the report). This was a real bug caught once already; this
test exists so it can't come back silently.
"""

import pytest

from score_aggregation import (
    chakkar_quality_score,
    mudra_identification_accuracy_score,
    mudra_rule_agreement_score,
    overall_score,
    timing_accuracy_score,
)


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


def test_full_orientation_reversal_scores_zero_on_that_subscore():
    """Real bug found and fixed: the docstring explicitly says a full 180
    degree orientation error should cost the rest of the 100 points, but
    the old formula (a flat /1.8 divisor) only reached ~8.3/100 at 180
    degrees -- a maximal, unambiguous orientation error still scored well
    above zero on this sub-score."""
    result = make_chakkar_result(count_gap=0.0, orientation_gap_deg=180.0, controlled_stop=True)
    score = chakkar_quality_score(result)
    # mean of (count_score=100, orientation_score=0, stop_score=100)
    assert score == pytest.approx(200.0 / 3.0)


def test_full_extra_rotation_scores_zero_on_that_subscore():
    """Same fix, the count-gap side: a full extra/missing rotation
    (count_gap == 1.0) should cost the rest of the 100 points too."""
    result = make_chakkar_result(count_gap=1.0, orientation_gap_deg=0.0, controlled_stop=True)
    score = chakkar_quality_score(result)
    assert score == pytest.approx(200.0 / 3.0)


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


def test_timing_accuracy_full_marks_when_perfectly_locked():
    windowed = [
        {"R": 1.0},
        {"R": 1.0},
    ]
    assert timing_accuracy_score(windowed) == 100.0


def test_timing_accuracy_score_is_none_when_no_windows():
    assert timing_accuracy_score([]) is None


def test_mudra_rule_agreement_score_is_self_consistency_not_accuracy():
    """Regression for the circularity an external review caught: a rule
    that's fully satisfied scores 100 even though this says nothing about
    whether the predicted mudra was the one actually intended."""
    assert mudra_rule_agreement_score([([], 4)]) == 100.0
    assert mudra_rule_agreement_score([(["a mismatch"], 4)]) == 75.0
    assert mudra_rule_agreement_score([]) is None
    assert mudra_rule_agreement_score([([], 0)]) is None  # no checkable constraints


def test_mudra_identification_accuracy_score_reflects_real_correctness():
    # 3 of 4 predictions matched the dancer's actual expected sequence.
    assert mudra_identification_accuracy_score([True, True, True, False]) == 75.0
    assert mudra_identification_accuracy_score([]) is None


def test_mudra_identification_and_rule_scores_can_disagree():
    """The exact scenario the circularity bug produces: a confidently WRONG
    identification whose predicted mudra's own rule still checks out clean."""
    rule_score = mudra_rule_agreement_score([([], 5)])  # predicted mudra's shape looks perfect
    id_score = mudra_identification_accuracy_score([False])  # but it was the wrong mudra
    assert rule_score == 100.0
    assert id_score == 0.0


def test_timing_accuracy_reflects_mean_phase_concentration():
    # mirrors the real reference clip finding: moderate concentration
    # (R ~0.3-0.6) throughout, not perfect lock and not random.
    windowed = [{"R": r} for r in (0.4, 0.5, 0.35, 0.55, 0.45, 0.6)]
    score = timing_accuracy_score(windowed)
    expected = 100 * (0.4 + 0.5 + 0.35 + 0.55 + 0.45 + 0.6) / 6
    assert abs(score - expected) < 1e-6


def test_timing_accuracy_score_is_clamped_to_0_100():
    """
    Real bug found and fixed: this used to score a window's binary
    Bonferroni-corrected significance flag rather than its R value, which
    -- at the small per-window sample sizes windowed_sync_check() actually
    produces (n=5-25 events per 8s window) -- routinely failed to reach
    significance even when R showed real, moderate phase-locking (R
    0.3-0.6, well above the ~1/sqrt(n) chance floor for these n). A real
    uploaded video with mean R ~0.45 scored 15.7% under the old rule
    despite genuine beat-locking through most of the clip. Scoring off R
    directly fixes that; this test just confirms the result still can't
    leave [0, 100] (R is bounded in [0, 1] but this guards the mapping).
    """
    perfectly_locked = [{"R": 1.0}]
    assert timing_accuracy_score(perfectly_locked) == 100.0

    perfectly_random = [{"R": 0.0}]
    assert timing_accuracy_score(perfectly_random) == 0.0
