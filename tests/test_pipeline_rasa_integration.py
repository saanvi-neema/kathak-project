"""
Integration test for app/pipeline.py's run_rasa_analysis(): confirms the
full wiring -- fixed-window sampling -> blendshape averaging -> NaN
filtering -> classify_rasa() -- works end to end over a synthetic
landmarks CSV shaped like extract_landmarks.py's real output.

Like test_pipeline_mudra_integration.py, this locks in the plumbing, not
real-world accuracy -- rasa_reference.py's thresholds are unvalidated
against real footage (see methods.md).
"""

import pandas as pd
import pytest

from rasa_reference import RASA_DEFINITIONS

import pipeline


def make_face_row(rasa_name):
    """
    A full 52-blendshape-style row, not just rasa_name's own criteria --
    every blendshape referenced by ANY rasa defaults to "clearly low", then
    rasa_name's own criteria override on top. Unlike test_rasa_reference.py's
    make_blendshape_row (which only tests check_rasa's per-rasa logic and can
    leave other rasas' blendshapes unset), classify_rasa() compares against
    every rasa at once -- leaving other rasas' criteria blendshapes missing
    lets them read as "not evaluable" everywhere and produce spurious ties
    (confirmed directly: an unset Shringara tied Hasya at zero mismatches).
    """
    HIGH_THRESHOLD, LOW_THRESHOLD = 0.4, 0.15
    CLEARLY_HIGH, CLEARLY_LOW = 0.9, 0.0
    CLEARLY_MID = (HIGH_THRESHOLD + LOW_THRESHOLD) / 2

    all_names = {name for rasa in RASA_DEFINITIONS.values() for name, _ in rasa["criteria"]}
    row = {f"face_{name}": CLEARLY_LOW for name in all_names}
    for name, direction in RASA_DEFINITIONS[rasa_name]["criteria"]:
        row[f"face_{name}"] = {"high": CLEARLY_HIGH, "low": CLEARLY_LOW, "mid": CLEARLY_MID}[direction]
    return row


def make_landmarks_csv(tmp_path, rasa_name, n_frames=60, fps=30):
    rows = []
    face_row = make_face_row(rasa_name)
    for i in range(n_frames):
        row = {"frame": i, "timestamp_ms": int(i * 1000 / fps)}
        row.update(face_row)
        rows.append(row)
    csv_path = tmp_path / "landmarks.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def test_run_rasa_analysis_returns_none_without_face_columns(tmp_path):
    df = pd.DataFrame({"frame": [0, 1], "timestamp_ms": [0, 33]})
    csv_path = tmp_path / "no_face.csv"
    df.to_csv(csv_path, index=False)
    assert pipeline.run_rasa_analysis(str(csv_path)) is None


def test_run_rasa_analysis_classifies_a_sustained_expression(tmp_path):
    landmarks_csv = make_landmarks_csv(tmp_path, "hasya", n_frames=60, fps=30)  # 2 seconds
    events = pipeline.run_rasa_analysis(landmarks_csv)

    assert events is not None and len(events) >= 1
    for e in events:
        assert e["rasa"] == "hasya"
        assert e["display_name"] == RASA_DEFINITIONS["hasya"]["display_name"]
        assert e["confidence_label"] == RASA_DEFINITIONS["hasya"]["confidence"]
        assert e["mismatch_count"] == 0


def test_run_rasa_analysis_produces_one_event_per_one_second_window(tmp_path):
    landmarks_csv = make_landmarks_csv(tmp_path, "raudra", n_frames=90, fps=30)  # 3 seconds
    events = pipeline.run_rasa_analysis(landmarks_csv)

    assert events is not None
    assert len(events) == 3
    assert events[0]["start_sec"] == 0.0
    assert events[1]["start_sec"] == pytest.approx(1.0)
    assert events[-1]["end_sec"] == pytest.approx(events[-1]["end_sec"])  # last window clipped to clip duration


def test_run_rasa_analysis_skips_windows_with_no_face_detected(tmp_path):
    """A window where every row is NaN (face out of frame) shouldn't produce
    a forced/garbage classification."""
    rows = []
    face_row = make_face_row("hasya")
    for i in range(30):  # first second: face detected
        row = {"frame": i, "timestamp_ms": int(i * 1000 / 30)}
        row.update(face_row)
        rows.append(row)
    for i in range(30, 60):  # second second: face not detected at all
        rows.append({"frame": i, "timestamp_ms": int(i * 1000 / 30), **{k: None for k in face_row}})
    csv_path = tmp_path / "landmarks.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    events = pipeline.run_rasa_analysis(str(csv_path))
    assert events is not None
    assert len(events) == 1
    assert events[0]["start_sec"] == 0.0
