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
