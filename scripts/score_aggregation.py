"""
App layer: Overview tab scoring.

Turns the detailed analysis output (chakkar_scoring, mudra_reference,
beat_sync_check) into the aggregate percentages the dashboard mockup's
Overview tab shows (e.g. "Mudra Accuracy: 92%", "Overall Score: 87%").

Some numbers here are principled -- they fall directly out of what the
other modules already measure. Others (the specific penalty scaling in
chakkar_quality_score, and equal-weighting in overall_score) are reasonable
defaults chosen now, not values derived from data -- there's no "bad
performance" ground truth yet to calibrate them against. Revisit once one
exists.
"""

import numpy as np


def mudra_rule_agreement_score(mudra_check_results):
    """
    NOT identification accuracy -- this measures something narrower and
    easy to mistake for it. `run_mudra_analysis()` predicts a mudra with
    the classifier, then checks the SAME predicted label's own geometric
    rule against the observed hand shape. So this score answers "does the
    hand shape agree with whatever mudra the classifier guessed" -- a
    confidently WRONG guess (e.g. tripataka misread as ardhapataka) can
    still score high here, because it's only checking the guess against
    itself, never against what the dancer actually intended. That's a real
    circularity, caught in an external review, not a hypothetical -- see
    methods.md step 4. Use mudra_identification_accuracy_score() instead
    for anything claiming to measure whether the identification was RIGHT
    -- that one needs an actual expected-mudra sequence to compare against.

    mudra_check_results: list of (mismatches, constraints_checked) tuples,
    one per mudra instance checked in the video. `mismatches` is the list
    returned by mudra_reference.check_mudra(); `constraints_checked` is how
    many individual things that mudra's rule actually tests (so a mudra with
    fewer checkable constraints doesn't get unfairly penalized per-mismatch).
    Returns a 0-100 score, or None if no mudras were checked.
    """
    if not mudra_check_results:
        return None
    per_mudra_scores = []
    for mismatches, constraints_checked in mudra_check_results:
        if constraints_checked <= 0:
            continue
        per_mudra_scores.append(1.0 - len(mismatches) / constraints_checked)
    if not per_mudra_scores:
        return None
    return 100 * float(np.mean(per_mudra_scores))


def mudra_identification_accuracy_score(match_results):
    """
    The actual identification-accuracy score mudra_rule_agreement_score()
    can't provide -- this compares each predicted mudra against a real
    expected label (supplied by the user, e.g. a known practice sequence),
    not against itself. Only meaningful when expected labels exist; the
    dashboard must not show this as if it always applies.

    match_results: list of bools, one per mudra event that HAD an expected
    label to compare against (True = predicted label matched expected).
    Events with no expected label (index ran past the supplied sequence)
    should be excluded before calling this, not passed in as False.
    Returns a 0-100 score, or None if the list is empty.
    """
    if not match_results:
        return None
    return 100 * float(np.mean(match_results))


def count_checked_constraints(rule):
    """How many individual things a mudra_reference.py rule dict actually checks."""
    n = len(rule.get("extended", [])) + len(rule.get("curled", []))
    thumb_touches = rule.get("thumb_touches", [])
    n += len(thumb_touches) if isinstance(thumb_touches, list) else (1 if thumb_touches else 0)
    n += len(rule.get("spread_pairs", []))
    n += 1 if "thumb_spread_from" in rule else 0
    return n


def chakkar_quality_score(score_result, count_gap_threshold=0.05, orientation_threshold_deg=15,
                           uncontrolled_stop_penalty_score=60):
    """
    score_result: output of chakkar_scoring.score_chakkar() for one chakkar
    sequence. Returns a 0-100 score combining count accuracy, ending
    orientation, and stop quality.

    Deviations below the same thresholds report_assembly.py uses to decide
    whether something is even worth flagging score as a perfect 100 --
    matches the recorded feedback that a performance which reads as "clean"
    to a dancer's own eye shouldn't lose points for sub-perceptible
    measurement noise. Points only start coming off once a deviation is
    large enough that it would actually generate a flag; from there, the
    exact scaling (a full rotation or a full 180 degrees costing the rest
    of the 100 points) is still a reasonable default, not derived from data.
    """
    count_gap = abs(score_result["count_gap"])
    if count_gap <= count_gap_threshold:
        count_score = 100.0
    else:
        count_score = max(0.0, 100 - (count_gap - count_gap_threshold) * 100)

    orientation_gap = abs(score_result["orientation_gap_deg"])
    if orientation_gap <= orientation_threshold_deg:
        orientation_score = 100.0
    else:
        orientation_score = max(0.0, 100 - (orientation_gap - orientation_threshold_deg) / 1.8)

    stop_score = 100.0 if score_result["controlled_stop"] else uncontrolled_stop_penalty_score
    return float(np.mean([count_score, orientation_score, stop_score]))


def timing_accuracy_score(windowed_results, clip_duration_sec):
    """
    windowed_results: output of beat_sync_check.windowed_sync_check().
    Returns the % of the clip's duration NOT covered by a "not synchronized"
    window (merged for overlap, so double-counted overlapping windows don't
    understate the score).

    Clamped to [0, 100] -- a real bug an external review caught: without
    clamping, this could return a NEGATIVE score. windowed_sync_check()
    allows a window to end up to 1e-9s past clip_duration (its own
    floating-point tolerance), and this function trusts whatever
    windowed_results/clip_duration_sec it's given rather than re-deriving
    one from the other -- so unsynced_duration can exceed clip_duration_sec
    by a hair even in normal use, or by a lot if a caller ever passes
    windowed_results computed against a different duration than the one
    supplied here. A negative "accuracy" is nonsensical either way.
    """
    if not windowed_results:
        return None
    unsynced = [r for r in windowed_results if not r["synchronized"]]
    if not unsynced:
        return 100.0

    intervals = sorted((r["window_start"], r["window_end"]) for r in unsynced)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    unsynced_duration = sum(end - start for start, end in merged)
    score = 100 * (1 - unsynced_duration / clip_duration_sec)
    return max(0.0, min(100.0, score))


def overall_score(mudra_score=None, chakkar_score=None, timing_score=None):
    """Equal-weighted average of whichever category scores are available."""
    scores = [s for s in (mudra_score, chakkar_score, timing_score) if s is not None]
    if not scores:
        return None
    return float(np.mean(scores))
