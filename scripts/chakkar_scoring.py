"""
Phase 3 extension: Chakkar scoring (round-to-clean-landing, ending orientation,
stop-quality check).

Builds on chakkar_pilot.py's proven rotation-counting signal (validated:
exact match on 1/3/10-spin clips). This module turns a raw rotation count
into the actual feedback described in the target output, e.g.:
"you did about 3.75 spins instead of 4" / "did not end facing front".

Reads the *_angles.csv files chakkar_pilot.py already produces.

Usage:
    python chakkar_scoring.py --angles data/landmarks/chakkar_pilot/chakkar_03/chakkar_03_angles.csv
"""

import argparse

import numpy as np
import pandas as pd

CLEAN_LANDING_STEP = 0.5  # chakkars are choreographed to land on whole or half rotations
STOP_QUALITY_WINDOW_FRAC = 0.10  # fraction of clip used as the "final stretch" for the stop check
CONTROLLED_STOP_RATIO = 0.35  # final-stretch speed must drop below this fraction of peak speed


def unwrap_angles(df):
    known = df.dropna(subset=["shoulder_angle_deg"])
    frames = known["frame"].to_numpy()
    angles_deg = known["shoulder_angle_deg"].to_numpy()
    unwrapped = np.degrees(np.unwrap(np.radians(angles_deg)))
    return frames, angles_deg, unwrapped


def score_chakkar(angles_csv: str) -> dict:
    df = pd.read_csv(angles_csv)
    frames, wrapped, unwrapped = unwrap_angles(df)

    total_rotation_deg = unwrapped[-1] - unwrapped[0]
    raw_count = abs(total_rotation_deg) / 360.0

    # 1. Round to nearest clean landing, report the gap.
    rounded_count = round(raw_count / CLEAN_LANDING_STEP) * CLEAN_LANDING_STEP
    count_gap = raw_count - rounded_count

    # 2. Ending orientation vs. "front" -- front is defined as wherever the
    # dancer was facing at the start (mean of the first 10 known frames),
    # since that's the only reference available without external input.
    # Must average the UNWRAPPED angle, not the raw wrapped one -- a plain
    # mean of wrapped values is wrong whenever the window straddles the
    # +-180 discontinuity (e.g. [179.9, -179.9] naively averages near 0,
    # when the correct answer is ~180). Unwrapped has no such jump.
    front_ref = np.mean(unwrapped[:10])
    ending_ref = np.mean(unwrapped[-10:])
    orientation_gap = ((ending_ref - front_ref + 180) % 360) - 180  # wrap to [-180, 180]

    # 3. Stop quality: compare rotation speed in the final stretch of the
    # clip to the peak speed reached anywhere in the clip. A controlled,
    # intentional stop decelerates well before the clip ends; an abrupt
    # cutoff is still near-peak speed right up to the last frame.
    dt = np.gradient(frames.astype(float))
    dt[dt == 0] = np.nan
    angular_speed = np.abs(np.gradient(unwrapped) / dt)  # deg/frame

    n = len(angular_speed)
    window = max(3, int(n * STOP_QUALITY_WINDOW_FRAC))
    peak_speed = np.nanmax(angular_speed)
    final_speed = np.nanmean(angular_speed[-window:])
    stop_ratio = final_speed / peak_speed if peak_speed > 0 else 0.0
    controlled_stop = stop_ratio < CONTROLLED_STOP_RATIO

    return {
        "raw_count": raw_count,
        "rounded_count": rounded_count,
        "count_gap": count_gap,
        "orientation_gap_deg": orientation_gap,
        "stop_ratio": stop_ratio,
        "controlled_stop": controlled_stop,
    }


# --- Multi-event segmentation -------------------------------------------
#
# score_chakkar() above scores rotation across a WHOLE clip as one event --
# correct for the 3 ground-truth pilot clips (one clean spin sequence each),
# wrong for a real piece with several separate chakkar phrases and other
# movement (arm gestures, footwork) in between. Unwrapping the whole clip
# still gives the mathematically correct NET rotation, but a single blended
# number is meaningless when it's actually several distinct choreographed
# events -- confirmed directly on movement_01.mov (a real Birju Maharaj
# piece excerpt): scoring it as one clip gave a falsely "clean" single
# result (raw_count=2.01, quality=100, no flags) despite containing several
# separate rotation bursts that individually weren't clean landings.
#
# Segmentation: compute the net rotation rate (deg/sec) in a sliding window,
# classify a window as "spinning" once that rate clears a threshold.
# Calibrated directly against real data, not guessed: on movement_01.mov,
# genuine chakkar bursts measured 150-400+ deg/s in every 0.5s window they
# covered, while ordinary arm-driven shoulder movement never exceeded ~100
# deg/s except right at a burst's edge. Validated the other direction too --
# rerunning this against the 3 ground-truth clips still finds exactly one
# segment each, with the same count/orientation/stop-quality results
# score_chakkar() already gets right.
#
# One more real finding from calibration: a single isolated window reading
# as "spinning" (one 0.1s slice hitting ~375 deg/s with normal values
# immediately before and after) turned out to be a one-frame tracking
# glitch, not a real rotation -- filtered out below by requiring a minimum
# sustained duration, not just one window over threshold.

SPIN_THRESHOLD_DEG_PER_SEC = 150
SEGMENT_WINDOW_SEC = 0.5
SEGMENT_STEP_SEC = 0.1
SEGMENT_MERGE_GAP_SEC = 0.3     # bridges brief within-burst dips (e.g. a hitch mid-spin)
SEGMENT_MIN_DURATION_SEC = 0.3  # drops single-frame tracking-glitch spikes
SEGMENT_SCORE_PAD_SEC = 0.25    # ~half a window -- windowed detection lags the true edges by about that much, so pad scoring bounds to avoid clipping the real wind-up/wind-down
MIN_SEGMENT_ROTATION_DEG = 180  # a segment must cover at least half a turn to count as a chakkar, not incidental fast-but-brief shoulder movement


def windowed_rotation_rate(times, unwrapped, window_sec=SEGMENT_WINDOW_SEC, step_sec=SEGMENT_STEP_SEC):
    """Net rotation rate (deg/sec) in sliding windows across the unwrapped angle signal."""
    centers, rates = [], []
    if len(times) < 2:
        return np.array(centers), np.array(rates)
    start = times[0]
    while start + window_sec <= times[-1]:
        end = start + window_sec
        mask = (times >= start) & (times <= end)
        if mask.sum() >= 2:
            seg_t, seg_a = times[mask], unwrapped[mask]
            rates.append((seg_a[-1] - seg_a[0]) / (seg_t[-1] - seg_t[0]))
            centers.append((start + end) / 2)
        start += step_sec
    return np.array(centers), np.array(rates)


def segment_rotation_bursts(times, unwrapped, threshold_deg_per_sec=SPIN_THRESHOLD_DEG_PER_SEC,
                             window_sec=SEGMENT_WINDOW_SEC, step_sec=SEGMENT_STEP_SEC,
                             merge_gap_sec=SEGMENT_MERGE_GAP_SEC, min_duration_sec=SEGMENT_MIN_DURATION_SEC):
    """
    Finds distinct sustained-rotation events in a shoulder-angle time series.
    Returns a list of {"start": t, "end": t} dicts, one per candidate
    chakkar burst (still to be filtered by rotation amount when scored) --
    NOT one aggregate number for the whole clip.
    """
    centers, rates = windowed_rotation_rate(times, unwrapped, window_sec, step_sec)
    if len(centers) == 0:
        return []

    spinning = np.abs(rates) >= threshold_deg_per_sec
    raw_segments = []
    in_segment = False
    seg_start = None
    for i, is_spin in enumerate(spinning):
        if is_spin and not in_segment:
            seg_start = centers[i]
            in_segment = True
        if not is_spin and in_segment:
            raw_segments.append((seg_start, centers[i - 1]))
            in_segment = False
    if in_segment:
        raw_segments.append((seg_start, centers[-1]))

    if not raw_segments:
        return []

    merged = [list(raw_segments[0])]
    for start, end in raw_segments[1:]:
        if start - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [{"start": s, "end": e} for s, e in merged if (e - s) >= min_duration_sec]


def score_chakkar_segment(df, start_sec, end_sec, pad_sec=SEGMENT_SCORE_PAD_SEC):
    """
    Scores one rotation segment using the same measures as score_chakkar()
    (count vs. clean landing, ending orientation, stop quality), but scoped
    to just that segment's own frames -- front/end orientation measured at
    the segment's own start/end, not the whole clip's. Kept as separate
    logic from score_chakkar() rather than refactored to share code, since
    that function's exact behavior is pinned by ground-truth regression
    tests and isn't worth the risk of an accidental change.

    Returns None if the segment doesn't clear MIN_SEGMENT_ROTATION_DEG --
    i.e. it was fast enough and long enough to pass segment_rotation_bursts'
    speed/duration filters, but didn't actually cover enough rotation to be
    a real chakkar (e.g. a brief fast partial turn, not a spin).
    """
    known = df.dropna(subset=["shoulder_angle_deg"])
    t_all = known["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    ang_all = known["shoulder_angle_deg"].to_numpy()
    unwrapped_all = np.degrees(np.unwrap(np.radians(ang_all)))

    mask = (t_all >= start_sec - pad_sec) & (t_all <= end_sec + pad_sec)
    t = t_all[mask]
    unwrapped = unwrapped_all[mask]
    if len(t) < 2:
        return None

    total_rotation_deg = unwrapped[-1] - unwrapped[0]
    if abs(total_rotation_deg) < MIN_SEGMENT_ROTATION_DEG:
        return None
    raw_count = abs(total_rotation_deg) / 360.0

    rounded_count = round(raw_count / CLEAN_LANDING_STEP) * CLEAN_LANDING_STEP
    count_gap = raw_count - rounded_count

    edge_n = min(10, max(2, len(t) // 4))
    front_ref = np.mean(unwrapped[:edge_n])
    ending_ref = np.mean(unwrapped[-edge_n:])
    orientation_gap = ((ending_ref - front_ref + 180) % 360) - 180

    dt = np.gradient(t)
    dt[dt == 0] = np.nan
    angular_speed = np.abs(np.gradient(unwrapped) / dt)
    window = max(3, int(len(angular_speed) * STOP_QUALITY_WINDOW_FRAC))
    peak_speed = np.nanmax(angular_speed)
    final_speed = np.nanmean(angular_speed[-window:])
    stop_ratio = final_speed / peak_speed if peak_speed > 0 else 0.0
    controlled_stop = stop_ratio < CONTROLLED_STOP_RATIO

    return {
        "start_sec": float(t[0]),
        "end_sec": float(t[-1]),
        "raw_count": raw_count,
        "rounded_count": rounded_count,
        "count_gap": count_gap,
        "orientation_gap_deg": orientation_gap,
        "stop_ratio": stop_ratio,
        "controlled_stop": controlled_stop,
    }


def score_chakkar_events(angles_csv: str) -> list:
    """
    Full multi-event pipeline: load angles, segment into distinct rotation
    bursts, score each independently. Returns a list of score dicts (one per
    real chakkar event, each with its own start_sec/end_sec) -- empty list
    if the clip has no segment that clears the speed/duration/rotation
    filters, rather than faking one aggregate result for a clip with no real
    chakkar in it.
    """
    df = pd.read_csv(angles_csv)
    known = df.dropna(subset=["shoulder_angle_deg"])
    t = known["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    ang = known["shoulder_angle_deg"].to_numpy()
    unwrapped = np.degrees(np.unwrap(np.radians(ang)))

    segments = segment_rotation_bursts(t, unwrapped)
    results = []
    for seg in segments:
        scored = score_chakkar_segment(df, seg["start"], seg["end"])
        if scored is not None:
            results.append(scored)
    return results


def describe(result: dict) -> str:
    lines = []
    if abs(result["count_gap"]) < 0.05:
        lines.append(f"Chakkar count: {result['rounded_count']:.1f} spins (clean).")
    else:
        direction = "short of" if result["count_gap"] < 0 else "over"
        lines.append(
            f"Chakkar count: about {result['raw_count']:.2f} spins, "
            f"{abs(result['count_gap']):.2f} rotations {direction} a clean {result['rounded_count']:.1f}."
        )

    if abs(result["orientation_gap_deg"]) < 15:
        lines.append("Ended facing front (within tolerance).")
    else:
        side = "left" if result["orientation_gap_deg"] > 0 else "right"
        lines.append(f"Did not end facing front -- off by about {abs(result['orientation_gap_deg']):.0f} degrees to the {side}.")

    if result["controlled_stop"]:
        lines.append("Stop looked controlled (smooth deceleration).")
    else:
        lines.append(f"Stop looked abrupt/interrupted (still at {result['stop_ratio']*100:.0f}% of peak speed at the end) -- treat the count gap as a likely real error, not noise.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score a chakkar clip's rotation quality")
    parser.add_argument("--angles", required=True, help="Path to a *_angles.csv from chakkar_pilot.py")
    args = parser.parse_args()

    result = score_chakkar(args.angles)
    print(describe(result))
    print()
    print(result)


if __name__ == "__main__":
    main()
