"""
Phase 5: Report Assembly.

Takes the outputs the other analysis modules already computed (chakkar
scoring, beat-sync checking, mudra checking) and turns them into one sorted,
timestamped, plain-English list -- the actual final output described in
methods.md's target example:

    "At 0:42, your chakkar did not end facing front, you did about 3.75
    spins instead of 4. At 1:10, your movements were not on beat."

This module does no analysis of its own -- it only formats results that
chakkar_scoring.py, beat_sync_check.py, and mudra_reference.py already
produced. No new signal processing here.
"""

from dataclasses import dataclass


@dataclass
class Flag:
    timestamp_sec: float
    category: str  # "chakkar", "mudra", "tatkaar", "timing"
    message: str

    def format(self):
        mins, secs = divmod(int(round(self.timestamp_sec)), 60)
        return f"At {mins}:{secs:02d}, {self.message}"


def flags_from_chakkar(score_result, end_time_sec, count_gap_threshold=0.05, orientation_threshold_deg=15):
    """Turn a chakkar_scoring.score_chakkar() result into report flags."""
    flags = []

    if abs(score_result["count_gap"]) >= count_gap_threshold:
        direction = "short of" if score_result["count_gap"] < 0 else "over"
        flags.append(Flag(
            end_time_sec, "chakkar",
            f"your chakkar was about {abs(score_result['count_gap']):.2f} rotations {direction} "
            f"a clean {score_result['rounded_count']:.1f} (measured {score_result['raw_count']:.2f})."
        ))

    if abs(score_result["orientation_gap_deg"]) >= orientation_threshold_deg:
        side = "left" if score_result["orientation_gap_deg"] > 0 else "right"
        flags.append(Flag(
            end_time_sec, "chakkar",
            f"your chakkar did not end facing front -- off by about "
            f"{abs(score_result['orientation_gap_deg']):.0f} degrees to the {side}."
        ))

    if not score_result["controlled_stop"]:
        flags.append(Flag(
            end_time_sec, "chakkar",
            "your chakkar's stop looked abrupt/interrupted, not a controlled landing."
        ))

    return flags


def flags_from_beat_sync(windowed_results, category="timing"):
    """
    Turn beat_sync_check.windowed_sync_check() results into report flags.
    Adjacent/overlapping "not synchronized" windows are merged into a single
    flag spanning the whole unsynced stretch, instead of one flag every
    couple of seconds for what's really one continuous issue.
    """
    flags = []
    unsynced = [r for r in windowed_results if not r["synchronized"]]
    if not unsynced:
        return flags

    groups = [[unsynced[0]]]
    for r in unsynced[1:]:
        if r["window_start"] <= groups[-1][-1]["window_end"]:
            groups[-1].append(r)
        else:
            groups.append([r])

    for group in groups:
        start, end = group[0]["window_start"], group[-1]["window_end"]
        center = (start + end) / 2
        flags.append(Flag(
            center, category,
            f"your movements were not clearly on beat from about {start:.0f}s to {end:.0f}s."
        ))

    return flags


def flags_from_mudra_checks(mudra_events):
    """
    mudra_events: list of (timestamp_sec, mudra_name, mismatches) tuples,
    where `mismatches` is the list returned by mudra_reference.check_mudra().
    """
    flags = []
    for t, name, mismatches in mudra_events:
        for m in mismatches:
            flags.append(Flag(t, "mudra", f"in {name}, {m}."))
    return flags


def assemble_report(*flag_lists):
    """Merge, sort by timestamp, and format all flags into the final report lines."""
    all_flags = [f for flags in flag_lists for f in flags]
    all_flags.sort(key=lambda f: f.timestamp_sec)
    return [f.format() for f in all_flags]


def print_report(*flag_lists):
    lines = assemble_report(*flag_lists)
    if not lines:
        print("No issues flagged -- clean performance.")
        return
    for line in lines:
        print(line)
