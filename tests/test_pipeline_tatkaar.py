"""
Tests for pipeline.run_tatkaar_analysis() -- single-video tatkaar (rhythmic
footwork) detection, pipeline step 7 (see methods.md). The actual dense-
stretch detection logic (comparison.find_dense_stretches) is already
thoroughly tested with synthetic onset arrays in test_comparison.py; these
tests cover the new wrapper around it -- _onset_signal's None-handling and
packaging stretches into the events shape this pipeline returns -- by
monkeypatching pipeline._onset_signal so no real video/audio file is needed,
same "bypass the file I/O round trip" approach test_pipeline_beat_grid_split.py
uses for _beat_grid_from_audio.
"""
import pipeline


def test_returns_none_when_no_audio_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_onset_signal", lambda video_path, wav_path: None)
    assert pipeline.run_tatkaar_analysis("fake.mp4", str(tmp_path)) is None


def test_returns_none_when_no_dense_stretch_found(tmp_path, monkeypatch):
    # A handful of scattered onsets -- not a sustained tatkaar-like run.
    monkeypatch.setattr(
        pipeline, "_onset_signal",
        lambda video_path, wav_path: {"onset_times": [0.5, 3.0, 7.2], "env": None, "env_times": None},
    )
    assert pipeline.run_tatkaar_analysis("fake.mp4", str(tmp_path)) is None


def test_packages_a_dense_stretch_into_an_event(tmp_path, monkeypatch):
    # 10 onsets, 0.2s apart -- a real tatkaar-shaped run (well past the
    # 8-onset/0.7s-gap thresholds find_dense_stretches uses).
    onsets = [i * 0.2 for i in range(10)]
    monkeypatch.setattr(
        pipeline, "_onset_signal",
        lambda video_path, wav_path: {"onset_times": onsets, "env": None, "env_times": None},
    )

    result = pipeline.run_tatkaar_analysis("fake.mp4", str(tmp_path))

    assert result is not None
    assert len(result["events"]) == 1
    e = result["events"][0]
    assert e["start_sec"] == 0.0
    assert e["end_sec"] == onsets[-1]
    assert e["strike_count"] == 10
    assert e["strikes_per_sec"] == 10 / onsets[-1]


def test_two_separated_dense_stretches_produce_two_events(tmp_path, monkeypatch):
    stretch_a = [i * 0.2 for i in range(9)]              # 0.0 -- 1.6
    stretch_b = [20.0 + i * 0.2 for i in range(9)]        # 20.0 -- 21.6, well past the 0.7s gap
    monkeypatch.setattr(
        pipeline, "_onset_signal",
        lambda video_path, wav_path: {"onset_times": stretch_a + stretch_b, "env": None, "env_times": None},
    )

    result = pipeline.run_tatkaar_analysis("fake.mp4", str(tmp_path))

    assert result is not None
    assert len(result["events"]) == 2
    assert result["events"][0]["start_sec"] == 0.0
    assert result["events"][1]["start_sec"] == 20.0
