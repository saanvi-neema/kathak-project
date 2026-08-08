"""
Comparison analysis: teacher-vs-student mode from methods.md's Comparison
tab scoping.

Two performances of the same piece won't be frame-synced -- different
starts, different pacing -- so they're aligned first, then compared.

Signal used for both alignment and event detection: audio onsets (a clap, a
foot stomp, a bell/ghungroo hit all produce an audible onset). This can't
tell a clap apart from a stomp -- that would need a dedicated sound
classifier, which doesn't exist here -- so results and messages describe
onsets generically as "movement sound(s)" rather than claiming to know the
specific type.

Two pieces, matching methods.md's original scoping:
1. Alignment -- fastdtw over the two tracks' onset-strength envelopes maps
   a moment in the teacher's timeline to the corresponding moment in the
   student's, even if the student is running slow/fast at that point.
2. Event comparison -- onsets are detected independently in each track,
   grouped into "actions" (a burst of onsets close together, e.g. two quick
   claps), then each teacher action's time window is mapped onto the
   student's timeline via the alignment and the onset counts are compared.

This module has no file I/O -- it operates on plain arrays so it's directly
testable with synthetic data. app/pipeline.py handles loading audio and
running librosa's onset detection before calling into this.
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

ACTION_GAP_SEC = 0.6  # onsets closer together than this count as one "action" (e.g. two quick claps)
MATCH_WINDOW_PAD_SEC = 0.4  # padding added around a mapped teacher action window when counting student onsets

# --- Dense rhythmic stretches (tatkaar) -----------------------------------
#
# cluster_onsets()/compare_actions() above assume a "action" is a handful of
# onsets close together, like two quick claps -- fine for occasional
# discrete gestures, wrong for tatkaar. A real tatkaar phrase is many bols
# in a row at roughly the same fast pace, so it collapses into ONE giant
# "action" under that logic, and if the student does an extra avartan the
# whole thing turns into one confused, meaningless flag (confirmed directly:
# a synthetic 1-avartan-vs-2-avartan test produced a single garbled
# "16 vs 31" mismatch instead of anything useful). Worse, feeding that
# length mismatch into fastdtw's alignment can distort the mapping for
# everything else in the piece.
#
# Fix: detect a sustained run of closely-spaced onsets as its own "dense
# stretch" up front, before any of the above runs. Compare it by onset
# COUNT/duration instead of matching individual bols (which fastdtw can't
# do reliably for a repeated cycle anyway -- see comparison.py's module
# docstring), and remove it from the envelope fed to fastdtw entirely so a
# length mismatch inside it can't warp the alignment of anything around it.
#
# These thresholds are reasoned defaults, not calibrated against real
# tatkaar audio (no recorded tatkaar footage exists in this project yet --
# see methods.md) -- the logic itself is validated with synthetic tests,
# but the exact cutoffs may need adjusting once real recordings exist.
DENSE_GAP_THRESHOLD_SEC = 0.7    # consecutive onsets closer than this can belong to the same dense run
MIN_DENSE_STRETCH_ONSETS = 8     # a run needs at least this many onsets to count as a real rhythmic stretch, not just a quick multi-clap gesture
DENSE_COUNT_TOLERANCE = 2        # paired dense stretches within this many onsets of each other count as "matched"


def find_dense_stretches(onset_times, gap_threshold=DENSE_GAP_THRESHOLD_SEC, min_onsets=MIN_DENSE_STRETCH_ONSETS):
    """
    Groups onset times into runs of sustained closely-spaced onsets (tatkaar-
    like), separately from cluster_onsets()'s short-burst grouping. Returns
    a list of {"start", "end", "count"} dicts, one per qualifying run.
    """
    onset_times = sorted(onset_times)
    if len(onset_times) < min_onsets:
        return []
    runs = [[onset_times[0]]]
    for t in onset_times[1:]:
        if t - runs[-1][-1] <= gap_threshold:
            runs[-1].append(t)
        else:
            runs.append([t])
    return [{"start": r[0], "end": r[-1], "count": len(r)} for r in runs if len(r) >= min_onsets]


def _remove_dense_regions(times, env, dense_stretches):
    """Drops envelope samples that fall inside any dense stretch, so fastdtw never sees them."""
    times = np.asarray(times)
    env = np.asarray(env)
    if not dense_stretches:
        return times, env
    mask = np.ones(len(times), dtype=bool)
    for d in dense_stretches:
        mask &= ~((times >= d["start"]) & (times <= d["end"]))
    return times[mask], env[mask]


def _is_in_any(t, stretches):
    return any(d["start"] <= t <= d["end"] for d in stretches)


def _describe_repetition(ratio):
    """Phrase a dense-stretch count mismatch as a likely repeat count when the ratio is close to a whole number."""
    nearest_whole = round(ratio)
    if nearest_whole >= 2 and abs(ratio - nearest_whole) < 0.2:
        return f"looks like about {nearest_whole}x the teacher's -- possibly {nearest_whole - 1} extra repetition(s), not just a timing difference"
    return "a different count, not just a timing difference"


def compare_dense_stretches(teacher_dense, student_dense):
    """
    Pairs dense stretches by order (1st teacher stretch vs. 1st student
    stretch, etc.) and compares each pair by onset count/duration rather
    than matching individual onsets. Any stretches on one side with no
    counterpart on the other (e.g. the student had two tatkaar sections,
    the teacher only one) are reported separately as extras, not silently
    dropped or force-matched to the wrong thing.
    """
    paired = list(zip(teacher_dense, student_dense))
    comparisons = []
    for td, sd in paired:
        ratio = (sd["count"] / td["count"]) if td["count"] else None
        match = abs(sd["count"] - td["count"]) <= DENSE_COUNT_TOLERANCE
        comparisons.append({
            "teacher_time": td["start"], "teacher_end": td["end"], "teacher_count": td["count"],
            "student_time": sd["start"], "student_end": sd["end"], "student_count": sd["count"],
            "ratio": ratio, "match": match,
        })
    extra_teacher = teacher_dense[len(paired):]
    extra_student = student_dense[len(paired):]
    return comparisons, extra_teacher, extra_student


def flags_from_dense_comparisons(comparisons, extra_teacher, extra_student):
    from report_assembly import Flag
    flags = []
    for d in comparisons:
        if d["match"]:
            continue
        flags.append(Flag(
            d["teacher_time"], "comparison",
            f"in this tatkaar/rhythmic section, the teacher did {d['teacher_count']} strikes and you did "
            f"{d['student_count']} -- {_describe_repetition(d['ratio'])}."
        ))
    for extra in extra_teacher:
        flags.append(Flag(
            extra["start"], "comparison",
            f"the teacher had a rhythmic/tatkaar section here ({extra['count']} strikes) that doesn't seem to have a matching section in your video."
        ))
    for extra in extra_student:
        flags.append(Flag(
            extra["start"], "comparison",
            f"you had an extra rhythmic/tatkaar section here ({extra['count']} strikes) beyond what's in the teacher's video."
        ))
    return flags


def cluster_onsets(onset_times, gap_sec=ACTION_GAP_SEC):
    """Group onset times into actions: consecutive onsets within gap_sec of each other are one action."""
    onset_times = sorted(onset_times)
    if not onset_times:
        return []
    clusters = [[onset_times[0]]]
    for t in onset_times[1:]:
        if t - clusters[-1][-1] <= gap_sec:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return [{"start": c[0], "end": c[-1], "count": len(c)} for c in clusters]


TIME_FEATURE_WEIGHT = 0.3  # see align_envelopes docstring


def align_envelopes(teacher_env, student_env, time_weight=TIME_FEATURE_WEIGHT):
    """
    fastdtw over the two onset-strength envelopes. Returns the warping path:
    a list of (teacher_idx, student_idx) index pairs.

    Plain envelope-amplitude distance is ambiguous during silence: the long
    near-zero stretches between onsets have ~0 distance no matter how
    they're matched, so a pure-amplitude DTW path can drift arbitrarily
    within a quiet stretch and still be "optimal" -- confirmed directly on a
    synthetic tempo-shifted test, where a teacher onset ended up mapped to a
    totally different action's region. Each frame is instead represented as
    [normalized envelope value, small weight * normalized time position] --
    the position term breaks ties during silence in favor of a
    proportionally-similar-in-time alignment, without overriding a real
    envelope match when the audio content actually disagrees.
    """
    t = np.asarray(teacher_env, dtype=float)
    s = np.asarray(student_env, dtype=float)
    if len(t) == 0 or len(s) == 0:
        # Nothing left to align -- e.g. a whole clip (or its only audible
        # portion) was one sustained dense/tatkaar stretch, so
        # compare_performances() removed every envelope sample before this
        # was ever called (see _remove_dense_regions). np.max() on an empty
        # array raises ValueError; the caller's teacher/student "sparse"
        # onset lists are empty in exactly this situation too, so
        # compare_actions() never actually dereferences this empty path.
        return []
    t_norm = t / (np.max(t) or 1.0)
    s_norm = s / (np.max(s) or 1.0)
    t_pos = np.linspace(0, 1, len(t))
    s_pos = np.linspace(0, 1, len(s))
    t_feat = np.stack([t_norm, time_weight * t_pos], axis=1)
    s_feat = np.stack([s_norm, time_weight * s_pos], axis=1)
    _, path = fastdtw(t_feat, s_feat, dist=euclidean)
    return path


def _map_indices(teacher_idx, path):
    matches = [s_idx for t_idx, s_idx in path if t_idx == teacher_idx]
    if matches:
        return matches
    nearest = min(path, key=lambda p: abs(p[0] - teacher_idx))
    return [nearest[1]]


def map_time(t_time, teacher_frame_times, student_frame_times, path):
    """Map a teacher-timeline timestamp to the corresponding student timestamp via the DTW path."""
    teacher_frame_times = np.asarray(teacher_frame_times)
    t_idx = int(np.argmin(np.abs(teacher_frame_times - t_time)))
    student_indices = _map_indices(t_idx, path)
    return float(np.mean([student_frame_times[i] for i in student_indices]))


def compare_actions(teacher_onsets, student_onsets, teacher_frame_times, student_frame_times, path,
                     pad_sec=MATCH_WINDOW_PAD_SEC):
    """
    Core comparison: for each teacher "action" (a cluster of onsets), map
    its time window onto the student's timeline and count how many student
    onsets fall inside it. Returns a list of dicts, one per teacher action:
    {teacher_time, student_time, teacher_count, student_count, match}.
    """
    teacher_actions = cluster_onsets(teacher_onsets)
    student_onsets_sorted = np.array(sorted(student_onsets))
    teacher_frame_times = np.asarray(teacher_frame_times)
    student_frame_times = np.asarray(student_frame_times)

    results = []
    for action in teacher_actions:
        start_idx = int(np.argmin(np.abs(teacher_frame_times - action["start"])))
        end_idx = int(np.argmin(np.abs(teacher_frame_times - action["end"])))
        student_start_indices = _map_indices(start_idx, path)
        student_end_indices = _map_indices(end_idx, path)

        window_start = student_frame_times[min(student_start_indices)] - pad_sec
        window_end = student_frame_times[max(student_end_indices)] + pad_sec

        if len(student_onsets_sorted):
            in_window = student_onsets_sorted[
                (student_onsets_sorted >= window_start) & (student_onsets_sorted <= window_end)
            ]
        else:
            in_window = np.array([])

        mapped_time = float(np.mean([student_frame_times[i] for i in student_start_indices]))
        results.append({
            "teacher_time": action["start"],
            "student_time": mapped_time,
            "teacher_count": action["count"],
            "student_count": int(len(in_window)),
            "match": action["count"] == int(len(in_window)),
        })
    return results


def flags_from_comparison(comparisons):
    """Turn compare_actions() output into report_assembly.Flag objects for the mismatched actions."""
    from report_assembly import Flag
    flags = []
    for c in comparisons:
        if c["match"]:
            continue
        if c["teacher_count"] > c["student_count"]:
            msg = (f"the teacher had {c['teacher_count']} movement sound(s) here "
                   f"(e.g. a clap or foot stomp), you only had {c['student_count']}.")
        else:
            msg = (f"the teacher had {c['teacher_count']} movement sound(s) here, "
                   f"you had {c['student_count']} -- an extra sound or a timing slip.")
        flags.append(Flag(c["teacher_time"], "comparison", msg))
    return flags


def compare_performances(teacher_env, teacher_env_times, teacher_onsets,
                          student_env, student_env_times, student_onsets):
    """
    End-to-end comparison over already-computed onset-strength envelopes and
    onset event times for both tracks. Returns a dict with the per-action
    comparisons, dense-rhythmic-stretch (tatkaar) comparisons, a match-rate
    summary, and report flags.

    Dense stretches are found and compared first, then excluded from both
    the alignment envelope and the isolated-action comparison -- see the
    "Dense rhythmic stretches" section above for why.
    """
    teacher_dense = find_dense_stretches(teacher_onsets)
    student_dense = find_dense_stretches(student_onsets)

    t_env_times, t_env = _remove_dense_regions(teacher_env_times, teacher_env, teacher_dense)
    s_env_times, s_env = _remove_dense_regions(student_env_times, student_env, student_dense)
    path = align_envelopes(t_env, s_env)

    teacher_sparse = [t for t in teacher_onsets if not _is_in_any(t, teacher_dense)]
    student_sparse = [t for t in student_onsets if not _is_in_any(t, student_dense)]
    comparisons = compare_actions(teacher_sparse, student_sparse, t_env_times, s_env_times, path)

    dense_comparisons, extra_teacher_dense, extra_student_dense = compare_dense_stretches(teacher_dense, student_dense)

    matched = sum(1 for c in comparisons if c["match"]) + sum(1 for d in dense_comparisons if d["match"])
    total = len(comparisons) + len(dense_comparisons) + len(extra_teacher_dense) + len(extra_student_dense)
    match_rate = 100.0 * matched / total if total else None

    flags = flags_from_comparison(comparisons) + flags_from_dense_comparisons(
        dense_comparisons, extra_teacher_dense, extra_student_dense
    )
    flags.sort(key=lambda f: f.timestamp_sec)

    return {
        "actions": comparisons,
        "dense_sections": dense_comparisons,
        "extra_teacher_sections": extra_teacher_dense,
        "extra_student_sections": extra_student_dense,
        "match_rate": match_rate,
        "flags": flags,
    }
