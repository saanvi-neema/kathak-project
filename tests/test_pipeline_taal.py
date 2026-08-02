"""
Tests for pipeline._taal_flags_from_chakkar -- the pure, audio-free core of
wiring taal/sam checking into chakkar analysis. Uses synthetic beat grids
and synthetic chakkar_scoring-shaped events, not real audio (real audio
extraction is a thin, separately-untested wrapper around this, same
pattern as comparison.py's core functions vs. its file-I/O wrapper).
"""

import numpy as np
import pytest

from pipeline import _taal_flags_from_chakkar, TAAL_OFFSET_FLAG_THRESHOLD


def make_chakkar_event(end_sec, start_sec=None):
    return {"result": {"start_sec": start_sec if start_sec is not None else end_sec - 1.0, "end_sec": end_sec}}


def make_beat_grid(sam_time, interval, n_matras, n_cycles):
    total = n_matras * n_cycles
    return sam_time + np.arange(total) * interval


def test_chakkar_ending_on_sam_is_not_flagged():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    # a chakkar ending exactly on the 3rd sam (matra 32 -> index 32)
    event = make_chakkar_event(end_sec=float(beats[32]))
    events, flags = _taal_flags_from_chakkar([event], beats, "teentaal", sam_time)

    assert len(events) == 1
    assert abs(events[0]["beats_from_sam"]) < 1e-6
    assert flags == []


def test_chakkar_ending_clearly_off_sam_is_flagged():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    # 2 matras after a sam
    event = make_chakkar_event(end_sec=float(beats[34]))  # sam at index 32, +2 matras
    events, flags = _taal_flags_from_chakkar([event], beats, "teentaal", sam_time)

    assert events[0]["beats_from_sam"] == pytest.approx(2.0, abs=0.05)
    assert len(flags) == 1
    assert "after sam" in flags[0].message
    assert "2.00 beats" in flags[0].message or "2.0" in flags[0].message


def test_chakkar_ending_before_sam_is_flagged_with_before_direction():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    # 1.5 matras before the sam at index 32 (i.e. index 32 - 1.5 matras)
    query_time = float(beats[32]) - 1.5 * interval
    event = make_chakkar_event(end_sec=query_time)
    events, flags = _taal_flags_from_chakkar([event], beats, "teentaal", sam_time)

    assert events[0]["beats_from_sam"] == pytest.approx(-1.5, abs=0.05)
    assert len(flags) == 1
    assert "before sam" in flags[0].message


def test_offset_within_deadzone_is_not_flagged():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    small_offset_sec = (TAAL_OFFSET_FLAG_THRESHOLD * 0.5) * interval
    event = make_chakkar_event(end_sec=float(beats[32]) + small_offset_sec)
    events, flags = _taal_flags_from_chakkar([event], beats, "teentaal", sam_time)

    assert abs(events[0]["beats_from_sam"]) < TAAL_OFFSET_FLAG_THRESHOLD
    assert flags == []


def test_offset_just_past_deadzone_is_flagged():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    just_over_sec = (TAAL_OFFSET_FLAG_THRESHOLD + 0.05) * interval
    event = make_chakkar_event(end_sec=float(beats[32]) + just_over_sec)
    events, flags = _taal_flags_from_chakkar([event], beats, "teentaal", sam_time)

    assert len(flags) == 1


def test_multiple_events_only_bad_landings_flagged():
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=3)
    clean = make_chakkar_event(end_sec=float(beats[16]))  # exactly on a sam
    dirty = make_chakkar_event(end_sec=float(beats[32]) + 3 * interval)  # 3 beats late

    events, flags = _taal_flags_from_chakkar([clean, dirty], beats, "teentaal", sam_time)
    assert len(events) == 2
    assert len(flags) == 1
    assert flags[0].timestamp_sec == dirty["result"]["end_sec"]


def test_different_taal_gives_different_cycle_length():
    """Same beat grid, different taal (Jhaptaal, 10 matras) -- the sam-relative
    math should differ because the cycle length differs."""
    sam_time, interval = 2.0, 0.5
    beats = make_beat_grid(sam_time, interval, n_matras=16, n_cycles=4)  # dense enough grid for either taal
    end_sec = float(beats[10])  # matra 10 in a 16-matra cycle, but matra 0 (sam) in a 10-matra cycle

    events_teen, _ = _taal_flags_from_chakkar([make_chakkar_event(end_sec)], beats, "teentaal", sam_time)
    events_jhap, _ = _taal_flags_from_chakkar([make_chakkar_event(end_sec)], beats, "jhaptaal", sam_time)

    assert events_teen[0]["beats_from_sam"] == pytest.approx(-6.0, abs=0.05)  # 6 short of the next teentaal sam (16)
    assert abs(events_jhap[0]["beats_from_sam"]) < 0.05  # lands right on a jhaptaal sam (multiple of 10)
