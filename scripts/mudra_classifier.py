"""
Trains and evaluates a mudra classifier from build_mudra_training_data.py's
labeled dataset.

Features: reuses extract_features.py's compute_hand_features() (finger curl
angles, thumb-to-fingertip distances, adjacent-finger spread, hand
orientation) -- the same geometric measurements mudra_reference.py's rules
already check by hand, just handed to a classifier instead of fixed
thresholds.

Model: RandomForestClassifier (scikit-learn, already a dependency) -- a
reasonable default for a small tabular dataset with no GPU/deep-learning
infrastructure here. Not tuned against real data, since none exists yet;
expect to revisit once a real dataset and its confusion matrix exist.

Usage:
    python mudra_classifier.py
    python mudra_classifier.py --dataset data/mudra_training/dataset.csv --model-out data/mudra_training/model.joblib
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

from extract_features import compute_hand_features  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.25
MIN_FRAMES_PER_MUDRA = 10  # a mudra with fewer labeled frames than this gets dropped rather than silently trained on almost nothing


def build_feature_matrix(dataset_df):
    """
    Turns the raw per-frame landmark dataset into geometric feature rows.
    compute_hand_features() computes both "left_*" and "right_*" versions of
    every feature regardless of which hand is actually present in a given
    row -- only each row's own labeled hand_side columns are kept, and the
    side prefix is dropped so left- and right-hand rows land on the same
    feature names.
    """
    feat = pd.DataFrame(index=dataset_df.index)
    compute_hand_features(dataset_df, feat)

    feature_rows = []
    for side, group in dataset_df.groupby("hand_side"):
        prefix = f"hand_{side}_"
        side_cols = [c for c in feat.columns if c.startswith(prefix)]
        if not side_cols:
            continue
        sub = feat.loc[group.index, side_cols].copy()
        sub.columns = [c[len(prefix):] for c in sub.columns]
        feature_rows.append(sub)

    if not feature_rows:
        return pd.DataFrame(), pd.Series(dtype=str)

    X = pd.concat(feature_rows).sort_index()
    y = dataset_df.loc[X.index, "mudra"]
    return X, y


def train_and_evaluate(dataset_csv, model_out=None, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    df = pd.read_csv(dataset_csv)

    counts = df.groupby("mudra").size()
    too_few = counts[counts < MIN_FRAMES_PER_MUDRA]
    if len(too_few):
        print(f"Dropping mudras with fewer than {MIN_FRAMES_PER_MUDRA} labeled frames: {list(too_few.index)}")
        df = df[~df["mudra"].isin(too_few.index)]

    X, y = build_feature_matrix(df)
    if X.empty or y.nunique() < 2:
        raise ValueError("Not enough labeled data to train a classifier (need at least 2 mudras with enough frames each).")

    # A few NaN feature values are expected (e.g. a finger briefly occluded
    # mid-hold) -- mean-impute rather than drop the whole row over one bad joint.
    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("Labels:", labels)
    print(matrix)

    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump({"model": model, "feature_names": list(X.columns), "labels": labels}, model_out)
        print(f"\nSaved model to {model_out}")

    return model, report, matrix


def predict_mudra(model_bundle, feature_row):
    """
    feature_row: a dict or Series of feature-name -> value, already stripped
    of the hand_<side>_ prefix (same convention build_feature_matrix uses).
    Returns (predicted_label, confidence) or (None, None) if a required
    feature is entirely missing.
    """
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    row = pd.Series(feature_row).reindex(feature_names)
    if row.isna().all():
        return None, None
    row = row.fillna(row.mean())
    proba = model.predict_proba(row.to_frame().T)[0]
    best_idx = int(np.argmax(proba))
    return model.classes_[best_idx], float(proba[best_idx])


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a mudra classifier")
    parser.add_argument("--dataset", default="data/mudra_training/dataset.csv")
    parser.add_argument("--model-out", default="data/mudra_training/model.joblib")
    args = parser.parse_args()
    train_and_evaluate(args.dataset, args.model_out)


if __name__ == "__main__":
    main()
