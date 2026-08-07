"""
Tests for pipeline._beat_grid_from_audio -- the pure tempo/beat-detection
logic split out of _extract_beat_grid so a live capture session's in-memory
rolling audio buffer can reuse it directly (see app/live_pipeline.py),
without a redundant wav-file round trip. _extract_beat_grid's own
video-path-to-wav extraction step is unchanged and already exercised
indirectly by existing pipeline tests -- these are for the new split point.
"""
import numpy as np

from pipeline import _beat_grid_from_audio, MIN_DURATION_FOR_WINDOWED_CHECK

SR = 22050


def test_silence_returns_none():
    y = np.zeros(SR * 10, dtype=np.float32)
    assert _beat_grid_from_audio(y, SR, duration_sec=10.0) is None


def test_short_duration_returns_none_even_with_loud_audio():
    duration = MIN_DURATION_FOR_WINDOWED_CHECK - 1.0
    n = int(SR * duration)
    y = ((np.random.RandomState(0).rand(n).astype(np.float32) - 0.5) * 2)  # loud noise, well above MIN_AUDIO_RMS
    assert _beat_grid_from_audio(y, SR, duration_sec=duration) is None


def test_click_train_produces_a_beat_grid():
    bpm = 120.0
    period_sec = 60.0 / bpm
    duration = 12.0
    y = np.zeros(int(SR * duration), dtype=np.float32)
    click_times = np.arange(0, duration, period_sec)
    for ct in click_times:
        idx = int(ct * SR)
        y[idx:idx + 50] = 1.0  # a short loud click, not just a single sample

    result = _beat_grid_from_audio(y, SR, duration_sec=duration)

    assert result is not None
    tempo_bpm, beat_times = result
    assert tempo_bpm > 0
    assert len(beat_times) >= 4
