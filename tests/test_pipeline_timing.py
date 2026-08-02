"""
Tests for pipeline._flags_from_ending -- the pure, audio-free core of
flagging a performance that finished noticeably before/after the music's
last beat. Uses synthetic beat grids, not real audio (same split as
test_pipeline_taal.py's _taal_flags_from_chakkar tests).
"""

import numpy as np
import pytest

from pipeline import _flags_from_ending, ENDING_OFFSET_FLAG_THRESHOLD_SEC


def make_beat_grid(n_beats=20, period=0.5, start=0.0):
    return start + np.arange(n_beats) * period


def test_ending_exactly_on_last_beat_is_not_flagged():
    beats = make_beat_grid()
    flags = _flags_from_ending(float(beats[-1]), beats)
    assert flags == []


def test_ending_within_deadzone_is_not_flagged():
    beats = make_beat_grid()
    small_offset = ENDING_OFFSET_FLAG_THRESHOLD_SEC * 0.5
    flags = _flags_from_ending(float(beats[-1]) + small_offset, beats)
    assert flags == []


def test_ending_clearly_before_last_beat_is_flagged():
    beats = make_beat_grid()
    duration = float(beats[-1]) - 1.0
    flags = _flags_from_ending(duration, beats)
    assert len(flags) == 1
    assert flags[0].category == "timing"
    assert flags[0].timestamp_sec == duration
    assert "before the ending beat" in flags[0].message
    assert "1.00" in flags[0].message


def test_ending_clearly_after_last_beat_is_flagged():
    beats = make_beat_grid()
    duration = float(beats[-1]) + 1.5
    flags = _flags_from_ending(duration, beats)
    assert len(flags) == 1
    assert "after the ending beat" in flags[0].message
    assert "1.50" in flags[0].message


def test_empty_beat_grid_produces_no_flags():
    assert _flags_from_ending(10.0, np.array([])) == []
