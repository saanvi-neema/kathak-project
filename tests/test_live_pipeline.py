"""
Tests for the pure bookkeeping functions in app/live_pipeline.py: chunk
timestamp advancement, and the "finalize margin" merge logic that decides
when a chakkar segment / mudra hold / rasa window is safe to report as
finished rather than still growing at the live buffer's edge (see
live_pipeline.py's module docstring for why this exists). A lightweight
SimpleNamespace stands in for a real LiveSession -- these functions only
read/write a handful of attributes, and a real LiveSession's __init__ loads
real MediaPipe landmarkers, which these pure-logic tests don't need.
"""
from types import SimpleNamespace

import live_pipeline as lp


def make_session(**overrides):
    base = dict(
        duration_sec=0.0,
        chakkar_events=[],
        rasa_events={},
        mudra_events={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def chakkar_event(end_sec, flags=None, quality_score=90.0):
    return {
        "result": {"end_sec": end_sec},
        "quality_score": quality_score,
        "flags": flags if flags is not None else [],
    }


def mudra_event(start_sec, end_sec, hand_side="right", mudra="pataka", mismatches=None,
                 constraints_checked=3, expected_mudra=None, matches_expected=None):
    return {
        "start_sec": start_sec, "end_sec": end_sec, "hand_side": hand_side, "mudra": mudra,
        "confidence": 0.5, "mismatches": mismatches, "constraints_checked": constraints_checked,
        "expected_mudra": expected_mudra, "matches_expected": matches_expected,
    }


def rasa_event(start_sec, end_sec, rasa="vira"):
    return {
        "start_sec": start_sec, "end_sec": end_sec, "rasa": rasa,
        "display_name": rasa.capitalize(), "confidence_label": "high", "mismatch_count": 0,
    }


# ---- _next_chunk_timestamp_ms ----

def test_next_chunk_timestamp_first_frame_is_zero():
    assert lp._next_chunk_timestamp_ms(-1, effective_fps=30.0) == 0.0


def test_next_chunk_timestamp_advances_by_frame_delta():
    ts = lp._next_chunk_timestamp_ms(1000.0, effective_fps=25.0)
    assert ts == 1000.0 + (1000.0 / 25.0)


def test_next_chunk_timestamp_stays_correct_across_a_changing_fps():
    # A live session's effective fps can differ chunk to chunk (see
    # _decode_chunk_frames) -- advancing by a per-frame delta from the last
    # timestamp (not an absolute frame_idx/fps position) must stay
    # monotonic and reasonable even when fps changes between calls.
    ts = 0.0
    for fps in [30.0, 24.0, 60.0, 15.0]:
        next_ts = lp._next_chunk_timestamp_ms(ts, effective_fps=fps)
        assert next_ts > ts
        ts = next_ts


# ---- chakkar finalize margin ----

def test_chakkar_event_within_margin_of_buffer_edge_is_not_merged():
    session = make_session(duration_sec=10.0)
    result = {"events": [chakkar_event(end_sec=9.8)]}  # 0.2s from edge, margin is 0.55s
    lp._merge_chakkar_events(session, result)
    assert session.chakkar_events == []


def test_chakkar_event_past_margin_is_merged():
    session = make_session(duration_sec=10.0)
    ev = chakkar_event(end_sec=9.0)  # 1.0s from edge, clears the 0.55s margin
    result = {"events": [ev]}
    lp._merge_chakkar_events(session, result)
    assert session.chakkar_events == [ev]


def test_chakkar_events_already_merged_are_not_duplicated_on_a_later_tick():
    session = make_session(duration_sec=10.0)
    ev = chakkar_event(end_sec=9.0)
    lp._merge_chakkar_events(session, {"events": [ev]})

    # A later tick: the buffer has grown, and re-running the full analysis
    # returns the same already-finalized event again plus a new one that's
    # now also past the margin.
    session.duration_sec = 10.55
    ev2 = chakkar_event(end_sec=9.5)
    lp._merge_chakkar_events(session, {"events": [ev, ev2]})

    assert session.chakkar_events == [ev, ev2]


def test_rebuild_chakkar_summary_combines_finalized_events():
    session = make_session()
    session.chakkar_events = [
        chakkar_event(end_sec=1.0, flags=["flagA"], quality_score=80.0),
        chakkar_event(end_sec=2.0, flags=[], quality_score=100.0),
    ]
    summary = lp._rebuild_chakkar_summary(session)
    assert summary["event_count"] == 2
    assert summary["quality_score"] == 90.0
    assert summary["flags"] == ["flagA"]


def test_rebuild_chakkar_summary_none_when_no_finalized_events():
    assert lp._rebuild_chakkar_summary(make_session()) is None


# ---- rasa finalize logic ----

def test_rasa_partial_trailing_window_is_not_merged():
    # run_rasa_analysis clips a still-accumulating window's end_sec to the
    # current buffer duration -- here the [0,1) window has only reached
    # 0.6s of its own 1s width, so it isn't done yet.
    session = make_session(duration_sec=0.6)
    result = [rasa_event(0.0, 0.6)]
    lp._merge_rasa_events(session, result)
    assert session.rasa_events == {}


def test_rasa_full_window_is_merged():
    session = make_session(duration_sec=2.0)
    result = [rasa_event(0.0, 1.0)]
    lp._merge_rasa_events(session, result)
    assert 0.0 in session.rasa_events
    assert session.rasa_events[0.0]["rasa"] == "vira"


def test_rasa_face_lost_mid_session_does_not_break_later_finalization():
    # A window with no detected face is skipped entirely by
    # run_rasa_analysis, which would silently shift list positions across
    # ticks if events were matched up by index instead of by start_sec.
    session = make_session(duration_sec=3.0)
    # Window [1,2) never shows up in any tick's raw result (face lost there).
    result = [rasa_event(0.0, 1.0), rasa_event(2.0, 3.0)]
    lp._merge_rasa_events(session, result)
    assert set(session.rasa_events.keys()) == {0.0, 2.0}


# ---- mudra finalize logic ----

def test_mudra_hold_within_margin_is_not_merged():
    session = make_session(duration_sec=5.0)
    ev = mudra_event(3.0, 4.8)  # 0.2s from edge, margin is STILL_WINDOW_SEC (0.3s)
    lp._merge_mudra_events(session, [ev])
    assert session.mudra_events == {}


def test_mudra_hold_past_margin_is_merged():
    session = make_session(duration_sec=5.0)
    ev = mudra_event(3.0, 4.5)  # 0.5s from edge, clears the margin
    lp._merge_mudra_events(session, [ev])
    assert session.mudra_events[("right", 3.0)] == ev


def test_mudra_hold_still_growing_updates_in_place_once_finalized():
    session = make_session(duration_sec=5.0)
    ev_short = mudra_event(3.0, 4.5, mudra="pataka")
    lp._merge_mudra_events(session, [ev_short])

    # Next tick: the same hold is still going, now longer, and the buffer
    # has grown too.
    session.duration_sec = 6.0
    ev_longer = mudra_event(3.0, 5.5, mudra="pataka")
    lp._merge_mudra_events(session, [ev_longer])

    assert len(session.mudra_events) == 1
    assert session.mudra_events[("right", 3.0)]["end_sec"] == 5.5


def test_assemble_mudra_summary_none_when_empty():
    assert lp._assemble_mudra_summary(make_session()) is None


def test_assemble_mudra_summary_includes_flags_for_mismatches():
    session = make_session()
    ev = mudra_event(0.0, 1.0, mismatches=["thumb not touching index"], constraints_checked=2)
    session.mudra_events = {("right", 0.0): ev}
    summary = lp._assemble_mudra_summary(session)
    assert summary["events"] == [ev]
    assert len(summary["flags"]) == 1
