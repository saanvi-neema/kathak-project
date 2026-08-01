"""
Tests for mudra_classifier.py against a SYNTHETIC dataset (see
mudra_test_helpers.py). These validate the pipeline is wired up correctly --
feature extraction, training, evaluation, prediction all work end to end --
not that real mudras are actually distinguishable. That's a separate
question, checked manually against a real trained model in methods.md.
"""

import numpy as np
import pandas as pd
import pytest

from mudra_classifier import build_feature_matrix, train_and_evaluate, predict_mudra
from mudra_test_helpers import SHAPES, make_hand_frame, make_synthetic_dataset


def test_build_feature_matrix_produces_separable_features():
    df = make_synthetic_dataset(n_frames_per_class=20)
    X, y = build_feature_matrix(df)
    assert len(X) == len(df)
    assert set(y.unique()) == set(SHAPES.keys())
    # mushti (all curled) should read as much more curled than alapadma (all extended)
    mushti_curl = X.loc[y == "mushti", "index_curl_angle"].mean()
    alapadma_curl = X.loc[y == "alapadma", "index_curl_angle"].mean()
    assert mushti_curl < alapadma_curl


def test_train_and_evaluate_end_to_end(tmp_path):
    df = make_synthetic_dataset(n_frames_per_class=40)
    dataset_csv = tmp_path / "dataset.csv"
    df.to_csv(dataset_csv, index=False)
    model_out = tmp_path / "model.joblib"

    model, report, matrix = train_and_evaluate(str(dataset_csv), model_out=str(model_out))

    assert model_out.exists()
    # clearly-separable synthetic shapes should be near-perfectly classifiable
    accuracy = np.trace(matrix) / matrix.sum()
    assert accuracy > 0.9, f"expected high accuracy on separable synthetic classes, got {accuracy:.2f}\n{report}"


def test_predict_mudra_returns_correct_label_and_confidence(tmp_path):
    df = make_synthetic_dataset(n_frames_per_class=40)
    dataset_csv = tmp_path / "dataset.csv"
    df.to_csv(dataset_csv, index=False)
    model_out = tmp_path / "model.joblib"
    train_and_evaluate(str(dataset_csv), model_out=str(model_out))

    import joblib
    bundle = joblib.load(model_out)

    rng = np.random.default_rng(99)
    frame = make_hand_frame(SHAPES["mushti"]["extended"], SHAPES["mushti"]["x_offsets"], rng=rng)
    row_df = pd.DataFrame([{f"hand_left_{i}_x": frame[i][0] for i in range(21)} |
                           {f"hand_left_{i}_y": frame[i][1] for i in range(21)}])
    from extract_features import compute_hand_features
    feat = pd.DataFrame(index=row_df.index)
    compute_hand_features(row_df, feat)
    feature_row = {c[len("hand_left_"):]: v for c, v in feat.iloc[0].items() if c.startswith("hand_left_")}

    label, confidence = predict_mudra(bundle, feature_row)
    assert label == "mushti"
    assert 0.0 < confidence <= 1.0


def test_too_few_frames_per_mudra_is_dropped_not_crashed(tmp_path):
    df = make_synthetic_dataset(n_frames_per_class=40)
    # cripple one class down to almost nothing
    sparse = df[df["mudra"] == "soochi"].iloc[:2]
    df = pd.concat([df[df["mudra"] != "soochi"], sparse], ignore_index=True)
    dataset_csv = tmp_path / "dataset.csv"
    df.to_csv(dataset_csv, index=False)

    model, report, matrix = train_and_evaluate(str(dataset_csv))
    assert "soochi" not in model.classes_
