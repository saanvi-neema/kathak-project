"""
Tests for beat_sync_check.py. These mirror the manual controls run once
during development (random data must NOT read as synchronized; a
perfectly-timed signal must) -- turning that one-off validation into a
permanent check, plus the subdivision behavior that was added afterward
to catch double/triple-time footwork.
"""

import numpy as np
import pytest

from beat_sync_check import compute_phases, rayleigh_test, subdivision_rayleigh_test

SEED = 42


def make_beat_grid(n_beats=100, period=0.44, start=0.1):
    return start + np.arange(n_beats) * period


def test_random_events_are_not_synchronized():
    rng = np.random.default_rng(SEED)
    beat_times = make_beat_grid()
    random_events = rng.uniform(0, 40, 150)
    phases = compute_phases(random_events, beat_times)
    _, p_value = rayleigh_test(phases)
    assert p_value > 0.05, "random, unsynced events should not read as synchronized"


def test_events_exactly_on_beat_are_synchronized():
    beat_times = make_beat_grid()
    on_beat_events = beat_times[:-1] + 0.02  # tiny realistic jitter
    phases = compute_phases(on_beat_events, beat_times)
    R, p_value = rayleigh_test(phases)
    assert p_value < 0.001
    assert R > 0.9


def test_double_time_events_need_subdivision_search_to_detect():
    """A real bug found during development: events landing twice evenly per
    beat look UNsynchronized to a plain single-phase test, even though the
    pattern is tight and real. subdivision_rayleigh_test must catch it."""
    beat_times = make_beat_grid()
    double_time_events = []
    for i in range(len(beat_times) - 1):
        start, end = beat_times[i], beat_times[i + 1]
        double_time_events.append(start + 0.02)
        double_time_events.append(start + (end - start) / 2 + 0.02)
    phases = compute_phases(np.array(double_time_events), beat_times)

    _, plain_p = rayleigh_test(phases)
    best = subdivision_rayleigh_test(phases, max_subdivision=4)

    assert best["subdivision"] == 2
    assert best["p_value"] < 0.001
    assert best["p_value"] < plain_p, "subdivision search should do better than the plain single-phase test"


def test_subdivision_search_does_not_falsely_synchronize_random_data():
    """Trying more subdivisions increases the chance of a false positive by
    chance -- the Bonferroni correction exists to guard against that."""
    rng = np.random.default_rng(SEED)
    beat_times = make_beat_grid()
    random_events = rng.uniform(0, 40, 150)
    phases = compute_phases(random_events, beat_times)
    best = subdivision_rayleigh_test(phases, max_subdivision=4)
    assert best["p_value"] > 0.05


def test_compute_phases_drops_events_outside_beat_grid_coverage():
    beat_times = make_beat_grid(n_beats=10)
    events = np.array([-5.0, 1000.0])  # before first beat, way after last beat
    phases = compute_phases(events, beat_times)
    assert len(phases) == 0
