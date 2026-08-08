"""
Tests for comparison.py (the Comparison tab's teacher-vs-student alignment
and action-count comparison). Uses synthetic onset-strength envelopes built
from Gaussian bumps at known onset times, instead of real audio, so the
alignment and clustering logic can be checked against a known ground truth
directly -- same approach as test_beat_sync_check.py's synthetic controls.
"""

import numpy as np
import pytest

from comparison import align_envelopes, cluster_onsets, compare_performances, find_dense_stretches


def make_envelope(onset_times, duration, hop_sec=0.02, sigma=0.03):
    """A synthetic onset-strength envelope: a Gaussian bump at each onset time."""
    times = np.arange(0, duration, hop_sec)
    env = np.zeros_like(times)
    for t in onset_times:
        env += np.exp(-0.5 * ((times - t) / sigma) ** 2)
    return env, times


def test_cluster_onsets_groups_close_events():
    clusters = cluster_onsets([1.0, 1.2, 1.5, 5.0, 8.0, 8.3])
    assert [c["count"] for c in clusters] == [3, 1, 2]
    assert clusters[0]["start"] == 1.0
    assert clusters[0]["end"] == 1.5


def test_cluster_onsets_respects_gap_threshold():
    # exactly at the gap threshold -> merged; just over it -> separate
    at_gap = cluster_onsets([0.0, 0.6], gap_sec=0.6)
    over_gap = cluster_onsets([0.0, 0.61], gap_sec=0.6)
    assert len(at_gap) == 1
    assert len(over_gap) == 2


def test_identical_performances_all_match():
    onsets = [1.0, 3.0, 3.2, 6.0, 9.0]
    duration = 11.0
    env, times = make_envelope(onsets, duration)

    result = compare_performances(env, times, onsets, env, times, onsets)

    assert result["match_rate"] == 100.0
    assert result["flags"] == []
    assert len(result["actions"]) == 4  # [1.0], [3.0,3.2], [6.0], [9.0]
    assert all(a["match"] for a in result["actions"])


def test_missing_onset_in_student_is_flagged():
    teacher_onsets = [1.0, 3.0, 3.2, 6.0, 9.0]
    student_onsets = [1.0, 3.0, 6.0, 9.0]  # missing the second onset of the 3.0 cluster
    duration = 11.0
    teacher_env, t_times = make_envelope(teacher_onsets, duration)
    student_env, s_times = make_envelope(student_onsets, duration)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    assert result["match_rate"] < 100.0
    mismatched = [a for a in result["actions"] if not a["match"]]
    assert len(mismatched) == 1
    assert mismatched[0]["teacher_count"] == 2
    assert mismatched[0]["student_count"] == 1
    assert len(result["flags"]) == 1
    assert "teacher had 2" in result["flags"][0].message


def test_extra_onset_in_student_is_flagged():
    teacher_onsets = [1.0, 3.0, 6.0, 9.0]
    student_onsets = [1.0, 3.0, 3.2, 6.0, 9.0]  # extra onset near 3.0
    duration = 11.0
    teacher_env, t_times = make_envelope(teacher_onsets, duration)
    student_env, s_times = make_envelope(student_onsets, duration)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    mismatched = [a for a in result["actions"] if not a["match"]]
    assert len(mismatched) == 1
    assert mismatched[0]["teacher_count"] == 1
    assert mismatched[0]["student_count"] == 2
    assert "extra sound or a timing slip" in result["flags"][0].message


# --- Dense rhythmic stretches (tatkaar) -----------------------------------
#
# Real bug found when the user asked directly: "what if the student does 2
# avartans of tatkaar and the teacher does 1?" Before this fix, that
# produced one garbled, meaningless flag (confirmed manually: "teacher had
# 16, you had 31"), and warped the alignment for everything else in the
# piece. These tests pin the fix: dense stretches must be found and
# compared by count instead of matched bol-by-bol, and must not distort the
# alignment of unrelated events elsewhere in the piece.

def test_find_dense_stretches_separates_tatkaar_from_isolated_claps():
    # a 10-onset "tatkaar" run (0.4s apart) plus two isolated claps far away
    tatkaar = list(np.arange(10) * 0.4)
    claps = [20.0, 25.0]
    onsets = tatkaar + claps

    dense = find_dense_stretches(onsets)
    assert len(dense) == 1
    assert dense[0]["count"] == 10
    assert dense[0]["start"] == 0.0
    assert dense[0]["end"] == pytest.approx(3.6)


def test_a_few_quick_claps_are_not_a_dense_stretch():
    """A quick double/triple clap (handled by cluster_onsets already) must
    not also get swept up as a "dense stretch" -- it's too short."""
    quick_claps = [1.0, 1.3, 1.6]  # 3 onsets, well under MIN_DENSE_STRETCH_ONSETS
    assert find_dense_stretches(quick_claps) == []


def test_extra_avartan_is_flagged_by_count_not_garbled():
    """The exact scenario the user asked about: teacher does 1 avartan (16
    bols), student does 2 (32 bols), both then do 3 more well-spaced claps."""
    teacher_onsets = list(np.arange(16) * 0.5 + 0.25) + [10.0, 15.0, 20.0]
    student_onsets = list(np.arange(32) * 0.5 + 0.25) + [18.0, 23.0, 28.0]
    teacher_env, t_times = make_envelope(teacher_onsets, 21.0)
    student_env, s_times = make_envelope(student_onsets, 29.0)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    assert len(result["dense_sections"]) == 1
    dense = result["dense_sections"][0]
    assert dense["teacher_count"] == 16
    assert dense["student_count"] == 32
    assert dense["match"] is False
    assert dense["ratio"] == pytest.approx(2.0)

    # exactly one flag about the tatkaar section, phrased as a count/repetition issue
    dense_flags = [f for f in result["flags"] if "tatkaar" in f.message]
    assert len(dense_flags) == 1
    assert "16" in dense_flags[0].message and "32" in dense_flags[0].message


def test_dense_mismatch_does_not_distort_downstream_alignment():
    """The real fix, not just the flag wording: claps AFTER a tatkaar-length
    mismatch must still align and match correctly, not get dragged off by
    the length difference earlier in the piece."""
    teacher_onsets = list(np.arange(16) * 0.5 + 0.25) + [10.0, 15.0, 20.0]
    student_onsets = list(np.arange(32) * 0.5 + 0.25) + [18.0, 23.0, 28.0]
    teacher_env, t_times = make_envelope(teacher_onsets, 21.0)
    student_env, s_times = make_envelope(student_onsets, 29.0)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    assert len(result["actions"]) == 3
    assert all(a["match"] for a in result["actions"]), "claps after the tatkaar mismatch should still match correctly"
    # mapped times should land close to the real student clap times (18, 23, 28), not drift
    expected_student_times = [18.0, 23.0, 28.0]
    for action, expected in zip(result["actions"], expected_student_times):
        assert abs(action["student_time"] - expected) < 1.0


def test_extra_dense_section_with_no_counterpart_is_reported_separately():
    """If one side has a dense section the other doesn't have at all (not
    just a different count), it shouldn't be force-paired with something
    unrelated -- it should show up as an unmatched extra."""
    teacher_onsets = [5.0, 10.0]  # no tatkaar at all
    student_onsets = list(np.arange(10) * 0.4) + [15.0, 20.0]  # has a tatkaar section plus 2 claps
    teacher_env, t_times = make_envelope(teacher_onsets, 11.0)
    student_env, s_times = make_envelope(student_onsets, 21.0)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    assert result["dense_sections"] == []
    assert len(result["extra_student_sections"]) == 1
    assert result["extra_student_sections"][0]["count"] == 10
    assert any("extra rhythmic/tatkaar section" in f.message for f in result["flags"])


def test_tempo_shifted_student_still_aligns():
    """
    Same action pattern, but the student's version is stretched 1.25x
    (running slower throughout) -- fastdtw should still map each teacher
    action onto the correct (shifted) student action, so counts still match
    even though the raw timestamps don't line up 1:1.
    """
    teacher_onsets = [1.0, 3.0, 3.2, 6.0, 9.0]
    stretch = 1.25
    student_onsets = [t * stretch for t in teacher_onsets]
    teacher_env, t_times = make_envelope(teacher_onsets, 11.0)
    student_env, s_times = make_envelope(student_onsets, 11.0 * stretch)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    assert result["match_rate"] == 100.0
    assert result["flags"] == []
    # mapped student time should land near the actual stretched onset, not the raw teacher time
    for action in result["actions"]:
        assert abs(action["student_time"] - action["teacher_time"] * stretch) < 0.5


def test_align_envelopes_returns_empty_path_for_empty_input_instead_of_crashing():
    """np.max() on an empty array raises ValueError -- align_envelopes must
    special-case this rather than let it bubble up (see
    test_whole_clip_as_one_dense_stretch_does_not_crash for the real
    end-to-end scenario that reaches this)."""
    assert align_envelopes(np.array([]), np.array([1.0, 2.0])) == []
    assert align_envelopes(np.array([1.0, 2.0]), np.array([])) == []
    assert align_envelopes(np.array([]), np.array([])) == []


def test_whole_clip_as_one_dense_stretch_does_not_crash():
    """Real bug found and fixed: a footwork-drill-style clip that's entirely
    one sustained tatkaar phrase (a realistic input for this project's
    tatkaar feature) removes every envelope sample before alignment
    (_remove_dense_regions), leaving align_envelopes() nothing to work with.
    Before the fix this raised an unhandled ValueError instead of producing
    a graceful dense-stretch-only comparison."""
    teacher_onsets = list(np.arange(20) * 0.3)  # one long dense run, nothing sparse
    student_onsets = list(np.arange(20) * 0.3)
    teacher_env, t_times = make_envelope(teacher_onsets, 7.0)
    student_env, s_times = make_envelope(student_onsets, 7.0)

    result = compare_performances(teacher_env, t_times, teacher_onsets, student_env, s_times, student_onsets)

    assert result["actions"] == []
    assert len(result["dense_sections"]) == 1
    assert result["dense_sections"][0]["match"] is True
