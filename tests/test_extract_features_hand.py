"""
Test for extract_features.compute_hand_features()'s handling of untracked
(NaN) hand landmarks in the "_extended" boolean-ish columns. Real bug found
and fixed: `pip_angle > 160` on a NaN pip_angle (an untracked landmark --
occlusion/motion blur, realistic during fast Kathak hand gestures) evaluated
to plain False rather than staying NaN, silently reporting "curled" for a
frame with no real tracking data instead of leaving it missing. That matters
downstream in app/pipeline.py's run_mudra_analysis, which averages this
column over a held window and needs real NaN to correctly exclude untracked
frames from the majority vote, the same way it already does for the other
(continuous) hand-feature columns.
"""
import numpy as np
import pandas as pd

from extract_features import HAND_FINGERS, compute_hand_features


def make_hand_df(n_rows=2):
    """A DataFrame with every hand_left_<0..20>_x/y column, all populated
    with distinct, valid (non-NaN) coordinates so every finger's angle is
    computable by default -- individual cells get overwritten with NaN by
    the tests below to simulate an untracked landmark."""
    data = {}
    for idx in range(21):
        data[f"hand_left_{idx}_x"] = [0.1 * idx + 0.01 * r for r in range(n_rows)]
        data[f"hand_left_{idx}_y"] = [0.05 * idx + 0.02 * r for r in range(n_rows)]
    return pd.DataFrame(data)


def test_extended_column_is_a_real_angle_when_fully_tracked():
    df = make_hand_df()
    feat = pd.DataFrame(index=df.index)
    compute_hand_features(df, feat)

    mcp, pip, dip, tip = HAND_FINGERS["index"]
    assert not np.isnan(feat.loc[0, "hand_left_index_curl_angle"])
    assert feat.loc[0, "hand_left_index_extended"] in (0.0, 1.0)


def test_extended_column_stays_nan_when_a_landmark_is_untracked():
    df = make_hand_df()
    mcp, pip, dip, tip = HAND_FINGERS["index"]
    # Simulate an untracked PIP landmark on row 1 only -- row 0 stays fully tracked.
    df.loc[1, f"hand_left_{pip}_x"] = np.nan
    df.loc[1, f"hand_left_{pip}_y"] = np.nan

    feat = pd.DataFrame(index=df.index)
    compute_hand_features(df, feat)

    assert np.isnan(feat.loc[1, "hand_left_index_curl_angle"])
    assert np.isnan(feat.loc[1, "hand_left_index_extended"]), (
        "an untracked landmark must leave _extended as NaN, not silently become False/curled"
    )
    # row 0 (fully tracked) must be unaffected by row 1's missing data
    assert not np.isnan(feat.loc[0, "hand_left_index_extended"])


def test_window_average_correctly_skips_untracked_frames():
    """The actual downstream payoff: app/pipeline.py's run_mudra_analysis
    averages this column over a held window and thresholds at 0.5 -- an
    untracked frame must be excluded from that average, not silently count
    as a "curled" (0.0) vote and drag a genuinely-extended window below the
    threshold."""
    df = make_hand_df(n_rows=3)
    mcp, pip, dip, tip = HAND_FINGERS["index"]

    # Fully-tracked baseline: whatever compute_hand_features naturally
    # computes for rows 0 and 2 from make_hand_df's coordinates.
    feat_baseline = pd.DataFrame(index=df.index)
    compute_hand_features(df, feat_baseline)
    expected_mean = feat_baseline.loc[[0, 2], "hand_left_index_extended"].mean()

    # Now make row 1's PIP landmark untracked -- rows 0/2 are untouched.
    df.loc[1, f"hand_left_{pip}_x"] = np.nan
    df.loc[1, f"hand_left_{pip}_y"] = np.nan
    feat = pd.DataFrame(index=df.index)
    compute_hand_features(df, feat)

    mean_with_untracked_row = feat["hand_left_index_extended"].mean()
    assert mean_with_untracked_row == expected_mean, (
        "an untracked row should be skipped by mean(), not counted as 0.0 and pull the average down"
    )
