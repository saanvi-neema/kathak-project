"""
Tests for the beat_grid contract on run_timing_analysis/run_taal_analysis.

Real regression found and fixed: these two functions used to accept
beat_grid=None as "not supplied, extract it yourself from video_path" and
silently fall back to _extract_beat_grid internally. Once analyze_video()
started always computing one shared beat_grid up front and passing it to
both (see test_pipeline_tatkaar.py-adjacent history in methods.md), that
fallback could no longer distinguish "the caller didn't pass anything" from
"the caller already tried and failed" -- a video with no usable audio
signal silently re-ran the full ffmpeg+librosa extraction inside each
function instead of just accepting the already-known failure. Fixed by
removing the fallback and the now-dead video_path parameter entirely,
making beat_grid required and authoritative. These tests lock that in:
neither function may call _extract_beat_grid itself under any circumstance.
"""
import pipeline


def test_run_timing_analysis_does_not_re_extract_when_beat_grid_is_none(tmp_path, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(pipeline, "_extract_beat_grid", lambda *a, **kw: called.update(n=called["n"] + 1))

    features_csv = tmp_path / "features.csv"
    features_csv.write_text("timestamp_ms\n0\n")

    result = pipeline.run_timing_analysis(str(features_csv), 10.0, None)

    assert result is None
    assert called["n"] == 0


def test_run_taal_analysis_does_not_re_extract_when_beat_grid_is_none(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(pipeline, "_extract_beat_grid", lambda *a, **kw: called.update(n=called["n"] + 1))

    chakkar = {"events": [{"result": {"start_sec": 0.0, "end_sec": 1.0}}]}
    result = pipeline.run_taal_analysis(10.0, chakkar, "teentaal", 1.0, None)

    assert result is None
    assert called["n"] == 0


def test_analyze_video_extracts_the_beat_grid_exactly_once(tmp_path, monkeypatch):
    """The actual end-to-end guarantee: analyze_video() shares one
    extraction across both checks, not two (or, before the fix, three)."""
    import numpy as np
    import pandas as pd

    call_count = {"n": 0}
    real_extract_beat_grid = pipeline._extract_beat_grid

    def counting_extract(*a, **kw):
        call_count["n"] += 1
        return real_extract_beat_grid(*a, **kw)

    monkeypatch.setattr(pipeline, "_extract_beat_grid", counting_extract)
    monkeypatch.setattr(pipeline, "get_video_duration", lambda video_path: 10.0)
    monkeypatch.setattr(pipeline, "extract_landmarks", lambda video_path, output_dir: str(tmp_path / "landmarks.csv"))
    pd.DataFrame({"frame": [0], "timestamp_ms": [0]}).to_csv(tmp_path / "landmarks.csv", index=False)
    monkeypatch.setattr(pipeline, "extract_features", lambda csv, output_dir: str(tmp_path / "features.csv"))
    pd.DataFrame({"timestamp_ms": [0]}).to_csv(tmp_path / "features.csv", index=False)
    # Must have real events -- run_taal_analysis short-circuits before ever
    # reaching the beat_grid check if chakkar has none, which would make
    # this test pass trivially without exercising the fixed code path.
    fake_chakkar = {"events": [{"result": {"start_sec": 0.0, "end_sec": 1.0}}], "flags": [], "quality_score": 90.0}
    monkeypatch.setattr(pipeline, "run_chakkar_analysis", lambda csv: fake_chakkar)
    monkeypatch.setattr(pipeline, "run_mudra_analysis", lambda csv, expected_sequence=None: None)
    monkeypatch.setattr(pipeline, "run_rasa_analysis", lambda csv: None)
    monkeypatch.setattr(pipeline, "run_tatkaar_analysis", lambda video_path, work_dir: None)
    monkeypatch.setattr(pipeline, "generate_pose_overlay", lambda *a, **kw: False)
    # No usable audio at all -- extract_audio always fails -- the scenario
    # that used to trigger 3 extraction attempts (analyze_video's own call,
    # then run_timing_analysis's fallback, then run_taal_analysis's).
    monkeypatch.setattr(pipeline, "extract_audio", lambda video_path, out_wav: False)

    result = pipeline.analyze_video("fake.mp4", str(tmp_path), taal_name="teentaal", sam_time=1.0)

    assert call_count["n"] == 1
    assert result["timing"] is None
    assert result["taal"] is None
