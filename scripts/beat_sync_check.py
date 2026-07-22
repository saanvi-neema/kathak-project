"""
Phase 4/6: Movement-vs-beat synchronization check (phase-based, not distance-based).

Replaces the earlier "distance from movement peak to nearest beat in
milliseconds" approach, which was proven statistically meaningless (a random,
unsynced signal scores just as well when beats are closely spaced -- see
methods.md). This version measures WHERE within each beat-to-beat gap a
movement lands (0% = right on the beat, 50% = exactly between two beats),
then tests whether those positions cluster together (real syncing) or spread
out evenly (no syncing) using a Rayleigh test -- the standard statistical
test for "are these points on a circle clustered, or uniformly spread out."

Usage:
    from beat_sync_check import compute_phases, rayleigh_test
    phases = compute_phases(movement_times, beat_times)
    R, p_value = rayleigh_test(phases)
"""

import numpy as np


def compute_phases(event_times, beat_times):
    """
    For each event time, find which beat-interval [beat[i], beat[i+1]) it
    falls in, and return its position within that interval as a fraction
    in [0, 1). Events before the first beat or after the last beat are
    dropped (no enclosing interval to measure against).
    """
    beat_times = np.asarray(sorted(beat_times))
    phases = []
    for t in event_times:
        idx = np.searchsorted(beat_times, t, side="right") - 1
        if idx < 0 or idx >= len(beat_times) - 1:
            continue  # outside the beat grid's coverage
        interval_start = beat_times[idx]
        interval_len = beat_times[idx + 1] - beat_times[idx]
        if interval_len <= 0:
            continue
        phases.append((t - interval_start) / interval_len)
    return np.array(phases)


def rayleigh_test(phases):
    """
    Rayleigh test for circular non-uniformity. `phases` are values in [0, 1)
    (fraction through a cycle). Returns (R, p_value):
      R = mean resultant length, 0 (uniformly spread) to 1 (all identical).
      p_value = probability of seeing this much clustering if the phases
                were actually uniformly random. Small p (< 0.05) = real
                clustering, i.e. genuinely synchronized to the beat.
    """
    n = len(phases)
    if n < 2:
        return 0.0, 1.0

    angles = phases * 2 * np.pi
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n

    z = n * R**2
    # Zar (1999) approximation for the Rayleigh test p-value.
    p_value = np.exp(-z) * (
        1
        + (2 * z - z**2) / (4 * n)
        - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)
    )
    p_value = float(np.clip(p_value, 0.0, 1.0))

    return float(R), p_value


def mean_phase(phases):
    """Circular mean of the phases, in [0, 1) -- where movements tend to cluster, if they do."""
    angles = phases * 2 * np.pi
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return (mean_angle / (2 * np.pi)) % 1.0


def subdivision_rayleigh_test(phases, max_subdivision=4):
    """
    Like rayleigh_test(), but also checks whether events are synchronized to
    a rhythmic SUBDIVISION of the beat (2 strikes per beat, 3, 4, ...), not
    just landing once per beat. Fast tatkaar commonly hits multiple evenly-
    spaced times per beat -- a plain single-phase Rayleigh test would wrongly
    read that as "unsynchronized" because the strikes spread across more than
    one cluster point, even though the pattern is tight and real.

    For subdivision k, phase*k mod 1 maps k evenly-spaced cluster points onto
    one, so a genuinely k-per-beat pattern shows up as tight clustering at
    that k. Tries k=1..max_subdivision and keeps whichever fits best.

    Testing multiple k values and keeping the best result inflates the
    chance of a false positive (more rolls of the dice = more likely one
    looks significant by chance), so the reported p-value is Bonferroni-
    corrected: multiplied by max_subdivision, capped at 1.0.

    A genuine k-per-beat pattern is mathematically also a valid fit at any
    multiple of k (e.g. a perfect 2x/beat pattern also looks like a perfect
    4x/beat pattern, since 2 divides evenly into 4) -- so when multiple
    subdivisions fit comparably well, the smallest one is preferred as the
    simpler, more parsimonious description, rather than picking whichever
    has a marginally lower p-value by numerical coincidence.
    """
    if len(phases) < 2:
        return {"subdivision": 1, "R": 0.0, "p_raw": 1.0, "p_value": 1.0}

    results = []
    for k in range(1, max_subdivision + 1):
        sub_phases = (phases * k) % 1.0
        R, p = rayleigh_test(sub_phases)
        results.append({"subdivision": k, "R": R, "p_raw": p})

    best_p_raw = min(r["p_raw"] for r in results)
    TIE_TOLERANCE = 1e-9
    tied = [r for r in results if r["p_raw"] <= best_p_raw + TIE_TOLERANCE]
    best = min(tied, key=lambda r: r["subdivision"])
    best["p_value"] = min(1.0, best["p_raw"] * max_subdivision)
    return best


def windowed_sync_check(event_times, beat_times, clip_duration, window_sec=8.0, step_sec=2.0,
                         min_events=5, alpha=0.05, max_subdivision=4):
    """
    Run the synchronization check in overlapping time windows instead of once
    over the whole clip. Not all movement is meant to be on-beat (expressive/
    narrative gestures are intentionally free-flowing in Kathak) -- testing
    the whole clip as one block would wash out real synchronization in the
    rhythmic sections. Phases are still computed against the FULL beat grid
    (not a windowed one) so events near a window's edge aren't measured
    against an artificially truncated beat interval.

    Uses subdivision_rayleigh_test() per window so a window where the
    dancer sped up to double/triple-time footwork still reads as
    synchronized, instead of being wrongly flagged just because the strikes
    no longer land once-per-beat.

    Returns a list of per-window results, each also carrying the window's
    center time (for reporting "at X seconds, movements were on beat / not").
    """
    beat_times = np.asarray(sorted(beat_times))
    event_times = np.asarray(sorted(event_times))

    results = []
    t = 0.0
    while t + window_sec <= clip_duration + 1e-9:
        window_events = event_times[(event_times >= t) & (event_times < t + window_sec)]
        if len(window_events) >= min_events:
            phases = compute_phases(window_events, beat_times)
            if len(phases) >= min_events:
                best = subdivision_rayleigh_test(phases, max_subdivision=max_subdivision)
                synchronized = best["p_value"] < alpha
                results.append({
                    "window_start": t,
                    "window_end": t + window_sec,
                    "window_center": t + window_sec / 2,
                    "n": len(phases),
                    "subdivision": best["subdivision"],
                    "R": best["R"],
                    "p_value": best["p_value"],
                    "synchronized": synchronized,
                })
        t += step_sec
    return results


def describe(phases, alpha=0.05, max_subdivision=4):
    best = subdivision_rayleigh_test(phases, max_subdivision=max_subdivision)
    k, R, p = best["subdivision"], best["R"], best["p_value"]
    result = {"n": len(phases), "subdivision": k, "R": R, "p_value": p, "synchronized": p < alpha}
    if p < alpha:
        sub_phases = (phases * k) % 1.0
        result["mean_phase"] = mean_phase(sub_phases)
        pct = result["mean_phase"] * 100
        rate = "once" if k == 1 else f"{k} times evenly"
        print(f"Movements ARE synchronized to the beat (p={p:.4f}, R={R:.3f}, n={len(phases)}, "
              f"landing {rate} per beat).")
        print(f"Within that, they tend to land about {pct:.0f}% of the way through each interval "
              f"({'right on it' if pct < 10 or pct > 90 else 'consistently offset from it'}).")
    else:
        print(f"No evidence of synchronization to the beat at any subdivision tested "
              f"(best fit: {k}x/beat, p={p:.4f}, R={R:.3f}, n={len(phases)}) "
              f"-- movement timing looks statistically indistinguishable from random relative to this beat grid.")
    return result
