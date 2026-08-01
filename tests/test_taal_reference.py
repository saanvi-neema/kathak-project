"""
Tests for taal_reference.py: sanity-checks the reference data itself
(manually transcribed from a cited source -- easy to typo a vibhag count),
then validates the cycle-alignment math against synthetic beat grids with a
known ground truth.
"""

import numpy as np
import pytest

from taal_reference import TAAL_DEFINITIONS, build_cycle_map, beats_from_sam, _vibhag_for_matra


@pytest.mark.parametrize("name,taal", list(TAAL_DEFINITIONS.items()))
def test_vibhags_sum_to_matras(name, taal):
    assert sum(taal["vibhags"]) == taal["matras"], f"{name}: vibhags {taal['vibhags']} don't sum to {taal['matras']} matras"


@pytest.mark.parametrize("name,taal", list(TAAL_DEFINITIONS.items()))
def test_khali_vibhags_are_valid_indices(name, taal):
    n_vibhags = len(taal["vibhags"])
    for idx in taal.get("khali_vibhags", []):
        assert 0 <= idx < n_vibhags, f"{name}: khali vibhag index {idx} out of range (only {n_vibhags} vibhags)"


def test_teentaal_khali_lands_on_matra_9():
    """Known real fact (cited): Teentaal is tali on 1/5/13, khali on 9."""
    taal = TAAL_DEFINITIONS["teentaal"]
    vibhag_index, is_khali = _vibhag_for_matra(taal, 8)  # matra 9 is 0-indexed 8
    assert is_khali


def test_teentaal_sam_is_tali():
    taal = TAAL_DEFINITIONS["teentaal"]
    _, is_khali = _vibhag_for_matra(taal, 0)
    assert not is_khali


def test_rupak_sam_is_khali():
    """Real, unusual fact about Rupak (cited): it's the one common taal
    where sam falls on a khali vibhag, not tali."""
    taal = TAAL_DEFINITIONS["rupak"]
    _, is_khali = _vibhag_for_matra(taal, 0)
    assert is_khali


def make_beat_grid(sam_time, matra_interval, n_matras, n_cycles):
    total = n_matras * n_cycles
    return sam_time + np.arange(total) * matra_interval


def test_build_cycle_map_assigns_correct_matra_and_cycle():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=2)
    cycle_map = build_cycle_map(beats, "teentaal", sam_time)

    assert cycle_map[0]["cycle"] == 0 and cycle_map[0]["matra"] == 1
    assert cycle_map[15]["cycle"] == 0 and cycle_map[15]["matra"] == 16
    assert cycle_map[16]["cycle"] == 1 and cycle_map[16]["matra"] == 1
    # matra 9 (0-indexed 8) should be khali
    assert cycle_map[8]["is_khali"]
    assert cycle_map[0]["is_khali"] is False


def test_build_cycle_map_before_sam_gives_negative_cycle():
    sam_time, interval = 10.0, 0.5
    beats = np.array([sam_time - 2 * interval, sam_time - interval, sam_time])
    cycle_map = build_cycle_map(beats, "teentaal", sam_time)
    assert cycle_map[-1]["cycle"] == 0 and cycle_map[-1]["matra"] == 1
    assert cycle_map[0]["cycle"] == -1  # two matras before sam wraps into the previous cycle


def test_beats_from_sam_at_sam_is_zero():
    sam_time, interval = 5.0, 0.4
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    result = beats_from_sam(sam_time + 32 * interval, beats, "teentaal", sam_time)  # exactly on a later sam
    assert abs(result) < 1e-6


def test_beats_from_sam_near_end_of_cycle_wraps_to_negative():
    """A moment that's 1 matra before the NEXT sam should read as -1, not +15."""
    sam_time, interval = 5.0, 0.4
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    query_time = sam_time + 15 * interval  # one matra short of the next sam (at +16)
    result = beats_from_sam(query_time, beats, "teentaal", sam_time)
    assert result == pytest.approx(-1.0, abs=0.05)


def test_beats_from_sam_partway_reports_fractional_offset():
    sam_time, interval = 5.0, 0.4
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    query_time = sam_time + 0.5 * interval  # halfway between sam and matra 2
    result = beats_from_sam(query_time, beats, "teentaal", sam_time)
    assert result == pytest.approx(0.5, abs=0.05)


def test_unknown_taal_raises():
    with pytest.raises(ValueError):
        build_cycle_map([0, 1, 2], "not_a_real_taal", 0.0)
    with pytest.raises(ValueError):
        beats_from_sam(1.0, [0, 1, 2], "not_a_real_taal", 0.0)


def test_too_few_beats_returns_empty_or_none():
    assert build_cycle_map([1.0], "teentaal", 1.0) == []
    assert beats_from_sam(1.0, [1.0], "teentaal", 1.0) is None
