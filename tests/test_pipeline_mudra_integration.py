"""
Integration test for app/pipeline.py's run_mudra_analysis(): trains a small
SYNTHETIC model (not the real ~740MB external dataset, which isn't
available in a fresh checkout and is gitignored) and confirms the full
wiring -- held-window detection -> feature extraction -> classifier
prediction -> reference-rule correctness check -- works end to end.

This locks in the plumbing, not real-world accuracy on real mudra footage.
That was checked manually against this project's own mudra_01.csv landmarks
once a real model existed (see methods.md) -- worth knowing, but not
something a synthetic regression test can verify.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mudra_test_helpers import SHAPES, make_hand_frame
from mudra_classifier import train_and_evaluate

import pipeline


def make_landmarks_csv(tmp_path, shape_name, n_frames=30, side="right", fps=30):
    """A synthetic 'video landmarks' CSV: n_frames of one held, near-static hand shape, matching
    the schema app/pipeline.py's run_mudra_analysis expects (hand_<side>_<i>_x/y + frame + timestamp_ms)."""
    shape = SHAPES[shape_name]
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n_frames):
        frame = make_hand_frame(shape["extended"], shape["x_offsets"], rng=rng)
        row = {"frame": i, "timestamp_ms": int(i * 1000 / fps)}
        for j in range(21):
            row[f"hand_{side}_{j}_x"] = frame[j][0]
            row[f"hand_{side}_{j}_y"] = frame[j][1]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "landmarks.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def synthetic_model(tmp_path):
    from mudra_test_helpers import make_synthetic_dataset
    dataset_csv = tmp_path / "dataset.csv"
    make_synthetic_dataset(n_frames_per_class=40).to_csv(dataset_csv, index=False)
    model_path = tmp_path / "model.joblib"
    train_and_evaluate(str(dataset_csv), model_out=str(model_path))
    return str(model_path)


def test_run_mudra_analysis_returns_none_without_a_trained_model(tmp_path):
    landmarks_csv = make_landmarks_csv(tmp_path, "mushti")
    missing_model = tmp_path / "no_model_here.joblib"
    assert pipeline.run_mudra_analysis(landmarks_csv, model_path=str(missing_model)) is None


def test_run_mudra_analysis_identifies_a_held_shape_and_checks_it(tmp_path, synthetic_model):
    landmarks_csv = make_landmarks_csv(tmp_path, "mushti")
    events = pipeline.run_mudra_analysis(landmarks_csv, model_path=synthetic_model)

    assert events is not None and len(events) >= 1
    event = events[0]
    assert event["mudra"] == "mushti"
    assert event["hand_side"] == "right"
    assert 0.0 < event["confidence"] <= 1.0
    # mushti is a real mudra_reference.py rule (all curled) and the
    # synthetic shape is built to satisfy it exactly -- correctness check
    # should find no mismatches.
    assert event["mismatches"] == []
    assert event["constraints_checked"] > 0


def test_run_mudra_analysis_reports_mismatches_for_a_malformed_hold(tmp_path, synthetic_model):
    """Same 'mushti' classifier prediction, but the actual hand shape only
    half-curls -- the correctness check (not the classifier) should catch
    that, same as it would for a real dancer's imperfect mudra."""
    shape = SHAPES["mushti"]
    rng = np.random.default_rng(2)
    rows = []
    for i in range(30):
        # only the pinky wrongly extended -- close enough to real mushti
        # that the classifier should still confidently call it "mushti"
        # (verified directly: ~0.7 confidence), while the correctness check
        # still catches the one wrong finger. Curling ALL 4 fingers wrong
        # makes the pose genuinely ambiguous between mudras and the
        # classifier's best guess drops below MUDRA_MIN_CONFIDENCE entirely
        # -- a real, correct behavior, just not what this test is checking.
        broken_extended = dict(shape["extended"])
        broken_extended["pinky"] = True
        frame = make_hand_frame(broken_extended, shape["x_offsets"], rng=rng)
        row = {"frame": i, "timestamp_ms": int(i * 1000 / 30)}
        for j in range(21):
            row[f"hand_right_{j}_x"] = frame[j][0]
            row[f"hand_right_{j}_y"] = frame[j][1]
        rows.append(row)
    csv_path = tmp_path / "broken_landmarks.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    events = pipeline.run_mudra_analysis(str(csv_path), model_path=synthetic_model)
    assert events is not None and len(events) >= 1
    assert events[0]["mudra"] == "mushti"
    assert events[0]["mismatches"] == ["pinky should be curled but was extended"]


def test_run_mudra_analysis_without_expected_sequence_leaves_match_fields_none(tmp_path, synthetic_model):
    landmarks_csv = make_landmarks_csv(tmp_path, "mushti")
    events = pipeline.run_mudra_analysis(landmarks_csv, model_path=synthetic_model)
    assert events[0]["expected_mudra"] is None
    assert events[0]["matches_expected"] is None


def test_run_mudra_analysis_flags_a_correct_identification_against_expected_sequence(tmp_path, synthetic_model):
    landmarks_csv = make_landmarks_csv(tmp_path, "mushti")
    events = pipeline.run_mudra_analysis(landmarks_csv, model_path=synthetic_model, expected_sequence=["mushti"])
    assert events[0]["expected_mudra"] == "mushti"
    assert events[0]["matches_expected"] is True


def test_run_mudra_analysis_flags_a_wrong_identification_against_expected_sequence(tmp_path, synthetic_model):
    """The exact circularity scenario: the classifier's guess ('mushti')
    passes its own rule check every time, but that says nothing about
    whether it was actually the mudra the dancer intended ('pataka' here)."""
    landmarks_csv = make_landmarks_csv(tmp_path, "mushti")
    events = pipeline.run_mudra_analysis(landmarks_csv, model_path=synthetic_model, expected_sequence=["pataka"])
    assert events[0]["mismatches"] == []  # rule agreement still looks clean
    assert events[0]["expected_mudra"] == "pataka"
    assert events[0]["matches_expected"] is False  # but the identification was wrong


def test_attach_expected_mudras_leaves_trailing_events_unmatched_when_sequence_is_shorter():
    """More detected holds than the supplied expected sequence -- the extras
    shouldn't get force-paired with something that doesn't exist. Tested
    directly against the extracted pure helper rather than trying to
    synthesize real hand-motion data that produces multiple held windows
    (which the underlying detector doesn't reliably do from a single
    continuous static pose, regardless of frame count)."""
    events = [{"mudra": "mushti"}, {"mudra": "pataka"}, {"mudra": "tripataka"}]
    pipeline._attach_expected_mudras(events, ["mushti"])
    assert events[0]["expected_mudra"] == "mushti"
    assert events[0]["matches_expected"] is True
    for e in events[1:]:
        assert e["expected_mudra"] is None
        assert e["matches_expected"] is None


def test_attach_expected_mudras_handles_no_expected_sequence():
    events = [{"mudra": "mushti"}]
    pipeline._attach_expected_mudras(events, None)
    assert events[0]["expected_mudra"] is None
    assert events[0]["matches_expected"] is None
