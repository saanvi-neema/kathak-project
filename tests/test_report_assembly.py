"""
Tests for report_assembly.py: flag generation from chakkar/beat-sync results,
and the merge-adjacent-unsynced-windows logic (built specifically to avoid
spamming one flag every 2 seconds for what's really one continuous issue).
"""

from report_assembly import Flag, assemble_report, flags_from_beat_sync, flags_from_chakkar


def make_chakkar_result(count_gap=0.0, orientation_gap_deg=0.0, controlled_stop=True,
                          raw_count=1.0, rounded_count=1.0):
    return {
        "count_gap": count_gap,
        "orientation_gap_deg": orientation_gap_deg,
        "controlled_stop": controlled_stop,
        "raw_count": raw_count,
        "rounded_count": rounded_count,
    }


def test_clean_chakkar_produces_no_flags():
    result = make_chakkar_result(count_gap=0.001, orientation_gap_deg=1.0, controlled_stop=True)
    assert flags_from_chakkar(result, end_time_sec=5.0) == []


def test_count_gap_past_threshold_produces_a_flag():
    result = make_chakkar_result(count_gap=0.25, raw_count=3.75, rounded_count=4.0)
    flags = flags_from_chakkar(result, end_time_sec=42.0)
    assert len(flags) == 1
    assert flags[0].category == "chakkar"
    assert flags[0].timestamp_sec == 42.0
    assert "3.75" in flags[0].message or "0.25" in flags[0].message


def test_uncontrolled_stop_produces_its_own_flag():
    result = make_chakkar_result(controlled_stop=False)
    flags = flags_from_chakkar(result, end_time_sec=10.0)
    assert any("abrupt" in f.message or "interrupted" in f.message for f in flags)


def test_flag_formats_timestamp_as_minutes_seconds():
    flag = Flag(timestamp_sec=125, category="chakkar", message="test message")
    assert flag.format() == "At 2:05, test message"


def test_adjacent_unsynced_windows_merge_into_one_flag():
    """Mirrors the real tatkaar finding: 5 overlapping unsynced windows
    should produce ONE flag spanning the whole stretch, not 5 separate ones."""
    windowed = [
        {"window_start": 20, "window_end": 28, "synchronized": False},
        {"window_start": 22, "window_end": 30, "synchronized": False},
        {"window_start": 24, "window_end": 32, "synchronized": False},
        {"window_start": 0, "window_end": 8, "synchronized": True},
    ]
    flags = flags_from_beat_sync(windowed, category="timing")
    assert len(flags) == 1
    assert "20s to 32s" in flags[0].message


def test_non_adjacent_unsynced_windows_stay_separate():
    windowed = [
        {"window_start": 0, "window_end": 8, "synchronized": False},
        {"window_start": 30, "window_end": 38, "synchronized": False},
    ]
    flags = flags_from_beat_sync(windowed, category="timing")
    assert len(flags) == 2


def test_assemble_report_sorts_by_timestamp_across_categories():
    chakkar_flags = [Flag(30.0, "chakkar", "chakkar issue")]
    timing_flags = [Flag(5.0, "timing", "timing issue")]
    lines = assemble_report(chakkar_flags, timing_flags)
    assert lines[0] == "At 0:05, timing issue"
    assert lines[1] == "At 0:30, chakkar issue"


def test_assemble_report_empty_when_no_flags():
    assert assemble_report([], []) == []
