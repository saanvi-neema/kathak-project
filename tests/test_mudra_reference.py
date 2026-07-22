"""
Tests for mudra_reference.py's rule-checking logic, using synthetic feature
rows constructed to exactly match or violate a rule -- this tests the
CHECKING LOGIC itself, independent of whether real footage is available.
Includes a regression case for the real bugs found and fixed for
Mrigasheesh and Singhamukh (both had inverted extended/curled states).
"""

import pytest

from mudra_reference import MUDRA_RULES, check_mudra, check_mudra_variants


def make_feature_row(rule, side="right", touch_ok=True, spread_ok=True):
    """Build a synthetic feature row that exactly satisfies `rule`."""
    row = {}
    prefix = f"hand_{side}"
    for finger in rule.get("extended", []):
        row[f"{prefix}_{finger}_extended"] = True
    for finger in rule.get("curled", []):
        row[f"{prefix}_{finger}_extended"] = False

    thumb_touches = rule.get("thumb_touches", [])
    if isinstance(thumb_touches, str):
        thumb_touches = [thumb_touches]
    for finger in thumb_touches:
        row[f"{prefix}_thumb_to_{finger}_tip"] = 0.01 if touch_ok else 0.5

    for f1, f2 in rule.get("spread_pairs", []):
        row[f"{prefix}_{f1}_{f2}_spread"] = 0.5 if spread_ok else 0.01

    if "thumb_spread_from" in rule:
        finger = rule["thumb_spread_from"]
        row[f"{prefix}_thumb_to_{finger}_tip"] = 0.5 if spread_ok else 0.01

    return row


@pytest.mark.parametrize("mudra_name", list(MUDRA_RULES.keys()))
def test_a_correctly_formed_mudra_has_no_mismatches(mudra_name):
    rule = MUDRA_RULES[mudra_name]
    row = make_feature_row(rule)
    mismatches = check_mudra(row, mudra_name, side="right")
    assert mismatches == [], f"{mudra_name}: expected no mismatches, got {mismatches}"


def test_extended_finger_reported_as_curled_is_flagged():
    row = make_feature_row(MUDRA_RULES["pataka"])
    row["hand_right_index_extended"] = False  # break it: pataka needs index extended
    mismatches = check_mudra(row, "pataka", side="right")
    assert any("index" in m for m in mismatches)


def test_mrigasheesh_regression():
    """Real bug: this rule used to have all 5 fingers extended; correct
    definition is only pinky+thumb extended, others curled."""
    rule = MUDRA_RULES["mrigasheesh"]
    assert set(rule["extended"]) == {"pinky", "thumb"}
    assert set(rule["curled"]) == {"index", "middle", "ring"}


def test_singhamukh_regression():
    """Real bug: extended/curled and thumb-touch target were inverted."""
    rule = MUDRA_RULES["singhamukh"]
    assert set(rule["curled"]) == {"middle", "ring"}
    assert set(rule["extended"]) == {"index", "pinky"}
    assert set(rule["thumb_touches"]) == {"middle", "ring"}


def test_katakamukh_variants_each_have_a_rule():
    for variant in ["katakamukh_1", "katakamukh_2", "katakamukh_3"]:
        assert variant in MUDRA_RULES


def test_check_mudra_variants_picks_the_closest_matching_variant():
    row = make_feature_row(MUDRA_RULES["katakamukh_2"])
    best_variant, mismatches = check_mudra_variants(row, "katakamukh", side="right")
    assert best_variant == "katakamukh_2"
    assert mismatches == []


def test_unknown_mudra_name_raises():
    with pytest.raises(ValueError):
        check_mudra({}, "not_a_real_mudra")
