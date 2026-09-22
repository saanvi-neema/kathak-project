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
        # Scaled so a full extra/missing rotation (count_gap == 1.0) costs
        # the rest of the 100 points, per this function's own docstring.
        # Real bug found and fixed: the flat "* 100" scale actually zeroed
        # out around count_gap ~= 1.05, not 1.0 -- a maximal, unambiguous
        # full-rotation miscount still scored 5/100 instead of 0. Scaling by
        # the actual remaining range (1.0 - count_gap_threshold) makes the
        # formula reach exactly 0 at the documented reference point.
        count_score = max(0.0, 100 - (count_gap - count_gap_threshold) * (100.0 / (1.0 - count_gap_threshold)))

    orientation_gap = abs(score_result["orientation_gap_deg"])
    if orientation_gap <= orientation_threshold_deg:
        orientation_score = 100.0
    else:
        # Same fix as count_score above: scaled by the actual remaining
        # range (180 - orientation_threshold_deg) so a full 180-degree
        # orientation error scores exactly 0, matching the docstring. The
        # flat "/ 1.8" divisor only reached ~8.3/100 at 180 degrees, not 0.
        orientation_score = max(0.0, 100 - (orientation_gap - orientation_threshold_deg)
                                 * (100.0 / (180.0 - orientation_threshold_deg)))

    stop_score = 100.0 if score_result["controlled_stop"] else uncontrolled_stop_penalty_score
    return float(np.mean([count_score, orientation_score, stop_score]))


def timing_accuracy_score(windowed_results):
    """
    windowed_results: output of beat_sync_check.windowed_sync_check().
    Returns the average phase concentration (Rayleigh R -- continuous,
    0 = phases spread uniformly/randomly, 1 = perfectly beat-locked)
    across all evaluated windows, scaled to 0-100.

    Real bug found and fixed: this used to score against each window's
    binary "synchronized" flag (p < 0.05 on the Rayleigh test, Bonferroni-
    corrected x4 for the 4 tested subdivisions), converted into "% of clip
    duration not covered by an unsynchronized window." That flag only
    answers "is there enough evidence, at this sample size, to prove
    clustering at p<0.05" -- it does not mean "confirmed random." With the
    sample sizes windowed_sync_check() actually produces (n=5-25 events
    per 8s window) and a x4 penalty stacked on top, real, moderate
    phase-locking (R around 0.3-0.6 -- well above the ~1/sqrt(n) chance
    floor for these n) routinely failed to clear that bar and got scored
    as a full 0 for that window. A real uploaded video with R averaging
    ~0.45 across 45 windows scored 15.7% under the old rule despite
    genuine, moderate beat-locking through most of the clip -- absence of
    proof (of significance) was being treated as proof of absence (of
    sync). Using R directly sidesteps significance testing entirely: it's
    a direct, continuous measure of how tightly movement clusters around
    a fixed beat-phase, not a pass/fail on a p-value.
    """
    if not windowed_results:
        return None
    mean_r = float(np.mean([w["R"] for w in windowed_results]))
    return max(0.0, min(100.0, mean_r * 100.0))


def overall_score(mudra_score=None, chakkar_score=None, timing_score=None):
    """Equal-weighted average of whichever category scores are available."""
    scores = [s for s in (mudra_score, chakkar_score, timing_score) if s is not None]
    if not scores:
        return None
    return float(np.mean(scores))
