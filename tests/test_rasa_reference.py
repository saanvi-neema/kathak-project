"""
Tests for rasa_reference.py's rule-checking logic, using synthetic
blendshape rows constructed to exactly match or violate a rule -- same
approach as test_mudra_reference.py: this tests the CHECKING LOGIC itself,
independent of whether real rasa-labeled footage exists to validate the
thresholds against (it doesn't yet -- see rasa_reference.py's docstring).
"""

import pytest

from rasa_reference import (
    RASA_DEFINITIONS,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    check_rasa,
    classify_rasa,
)

# Comfortably on either side of HIGH_THRESHOLD/LOW_THRESHOLD, and squarely
# inside the "mid" band between them.
CLEARLY_HIGH = 0.9
CLEARLY_LOW = 0.0
CLEARLY_MID = (HIGH_THRESHOLD + LOW_THRESHOLD) / 2


def make_blendshape_row(rasa_name, break_index=None):
    """
    Build a synthetic blendshape dict that exactly satisfies every criterion
    for `rasa_name`. If break_index is given, that one criterion is instead
    set to violate its direction (used to test mismatch detection).
    """
    rasa = RASA_DEFINITIONS[rasa_name]
    row = {}
    for i, (name, direction) in enumerate(rasa["criteria"]):
        want = direction
        if i == break_index:
            # flip to something that fails the check
            want = {"high": "low", "low": "high", "mid": "high"}[direction]
        if want == "high":
            row[name] = CLEARLY_HIGH
        elif want == "low":
            row[name] = CLEARLY_LOW
        else:
            row[name] = CLEARLY_MID
    return row


@pytest.mark.parametrize("rasa_name", list(RASA_DEFINITIONS.keys()))
def test_a_correctly_formed_rasa_has_no_mismatches(rasa_name):
    row = make_blendshape_row(rasa_name)
    mismatches = check_rasa(row, rasa_name)
    assert mismatches == [], f"{rasa_name}: expected no mismatches, got {mismatches}"


@pytest.mark.parametrize("rasa_name", list(RASA_DEFINITIONS.keys()))
def test_breaking_one_criterion_is_flagged(rasa_name):
    row = make_blendshape_row(rasa_name, break_index=0)
    mismatches = check_rasa(row, rasa_name)
    broken_name = RASA_DEFINITIONS[rasa_name]["criteria"][0][0]
    assert any(broken_name in m for m in mismatches)


def test_every_rasa_has_a_confidence_and_description():
    for name, rasa in RASA_DEFINITIONS.items():
        assert rasa["confidence"] in {"low", "medium", "high"}
        assert rasa["description"]


def test_vira_is_explicitly_flagged_low_confidence():
    """Vira (heroism) is a genuinely weak fit for single-frame blendshapes
    -- steadiness/composure isn't a facial-muscle-activation signature the
    way anger or disgust are. This should stay honestly marked, not quietly
    upgraded to look as reliable as the others."""
    assert RASA_DEFINITIONS["vira"]["confidence"] == "low"


def test_missing_blendshape_is_not_treated_as_a_mismatch():
    # An empty row means "no data for any of these blendshapes" -- that's
    # missing data, not evidence against the rasa.
    mismatches = check_rasa({}, "hasya")
    assert mismatches == []


def test_face_prefix_is_stripped():
    row = make_blendshape_row("hasya")
    prefixed = {f"face_{k}": v for k, v in row.items()}
    assert check_rasa(prefixed, "hasya") == []


def test_mid_direction_rejects_values_at_or_above_high_threshold():
    row = make_blendshape_row("shanta")
    row["mouthSmileLeft"] = HIGH_THRESHOLD  # boundary: mid requires strictly below
    mismatches = check_rasa(row, "shanta")
    assert any("mouthSmileLeft" in m for m in mismatches)


def test_mid_direction_rejects_values_at_or_below_low_threshold():
    row = make_blendshape_row("shanta")
    row["mouthSmileLeft"] = LOW_THRESHOLD  # boundary: mid requires strictly above
    mismatches = check_rasa(row, "shanta")
    assert any("mouthSmileLeft" in m for m in mismatches)


def test_unknown_rasa_name_raises():
    with pytest.raises(ValueError):
        check_rasa({}, "not_a_real_rasa")


def test_classify_rasa_picks_a_zero_mismatch_match():
    row = make_blendshape_row("bibhatsa")
    best, mismatches, scores = classify_rasa(row)
    assert scores[best] == 0
    assert set(scores.keys()) == set(RASA_DEFINITIONS.keys())
