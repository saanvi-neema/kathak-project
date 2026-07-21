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


def mudra_accuracy_score(mudra_check_results):
    """
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


def count_checked_constraints(rule):
    """How many individual things a mudra_reference.py rule dict actually checks."""
    n = len(rule.get("extended", [])) + len(rule.get("curled", []))
    thumb_touches = rule.get("thumb_touches", [])
    n += len(thumb_touches) if isinstance(thumb_touches, list) else (1 if thumb_touches else 0)
    n += len(rule.get("spread_pairs", []))
    n += 1 if "thumb_spread_from" in rule else 0
    return n


def chakkar_quality_score(score_result, uncontrolled_stop_penalty_score=60):
    """
    score_result: output of chakkar_scoring.score_chakkar() for one chakkar
    sequence. Returns a 0-100 score combining count accuracy, ending
    orientation, and stop quality.

    Scaling choices (not derived from data): a full rotation of count error
    or a full 180-degree orientation error each cost the full 100 points;
    an uncontrolled stop costs a flat penalty rather than scaling with
    anything, since "controlled vs. not" is currently binary, not a measured
    continuum.
    """
    count_score = max(0.0, 100 - abs(score_result["count_gap"]) * 100)
    orientation_score = max(0.0, 100 - abs(score_result["orientation_gap_deg"]) / 1.8)
    stop_score = 100.0 if score_result["controlled_stop"] else uncontrolled_stop_penalty_score
    return float(np.mean([count_score, orientation_score, stop_score]))


def timing_accuracy_score(windowed_results, clip_duration_sec):
    """
    windowed_results: output of beat_sync_check.windowed_sync_check().
    Returns the % of the clip's duration NOT covered by a "not synchronized"
    window (merged for overlap, so double-counted overlapping windows don't
    understate the score).
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
    return 100 * (1 - unsynced_duration / clip_duration_sec)


def overall_score(mudra_score=None, chakkar_score=None, timing_score=None):
    """Equal-weighted average of whichever category scores are available."""
    scores = [s for s in (mudra_score, chakkar_score, timing_score) if s is not None]
    if not scores:
        return None
    return float(np.mean(scores))
