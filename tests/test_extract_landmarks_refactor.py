"""
Tests for the pure helpers factored out of extract_landmarks()'s per-frame
loop -- next_video_timestamp_ms and extract_frame_landmarks. Factored out
so a live capture session (app/live_pipeline.py) can reuse long-lived
landmarkers across many short chunks instead of recreating them (and
reloading the .task model bundles) per chunk. These tests confirm the
split didn't change behavior -- fake landmarker stand-ins are used instead
of real MediaPipe models, matching this repo's existing convention of
testing plumbing with synthetic data, not real model weights.
"""
import numpy as np

from extract_landmarks import next_video_timestamp_ms, extract_frame_landmarks


def test_next_video_timestamp_ms_advances_with_frame_index():
    assert next_video_timestamp_ms(0, fps=30.0, last_ts=-1) == 0
    assert next_video_timestamp_ms(30, fps=30.0, last_ts=999) == 1000


def test_next_video_timestamp_ms_monotonic_guard_forces_at_least_one_ms_advance():
    # A pathologically high fps could compute a ts_ms that doesn't advance
    # past the previous frame's -- the guard forces at least +1ms so two
    # frames never collide on the same timestamp.
    ts = next_video_timestamp_ms(1, fps=1_000_000.0, last_ts=5)
    assert ts == 6


class FakeLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class FakePoseResult:
    def __init__(self, landmarks=None):
        self.pose_landmarks = [landmarks] if landmarks is not None else []


class FakeHandedness:
    def __init__(self, label):
        self.category_name = label


class FakeHandResult:
    def __init__(self, hands=None):
        hands = hands or []  # list of (label, [FakeLandmark]*21)
        self.hand_landmarks = [h[1] for h in hands]
        self.handedness = [[FakeHandedness(h[0])] for h in hands]


class FakeBlendshape:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class FakeFaceResult:
    def __init__(self, blendshapes=None):
        self.face_blendshapes = [blendshapes] if blendshapes is not None else []


class FakeLandmarker:
    def __init__(self, result):
        self._result = result

    def detect_for_video(self, mp_image, ts_ms):
        return self._result


def _blank_frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_extract_frame_landmarks_row_shape_with_full_detections():
    pose_landmarks = [FakeLandmark(x=i * 0.01, y=i * 0.01, visibility=1.0) for i in range(33)]
    pose = FakeLandmarker(FakePoseResult(pose_landmarks))
    hand = FakeLandmarker(FakeHandResult([("Right", [FakeLandmark(x=0.1, y=0.2) for _ in range(21)])]))
    face = FakeLandmarker(FakeFaceResult([FakeBlendshape("mouthSmileLeft", 0.42)]))

    row = extract_frame_landmarks(pose, hand, face, _blank_frame(), frame_idx=7, ts_ms=233)

    assert row["frame"] == 7
    assert row["timestamp_ms"] == 233
    assert row["body_nose_x"] == 0.0
    assert row["body_left_shoulder_vis"] == 1.0
    assert row["hand_right_0_x"] == 0.1
    assert row["face_mouthSmileLeft"] == 0.42


def test_extract_frame_landmarks_low_visibility_body_point_is_none():
    pose_landmarks = [FakeLandmark(visibility=0.1) for _ in range(33)]
    pose = FakeLandmarker(FakePoseResult(pose_landmarks))
    hand = FakeLandmarker(FakeHandResult([]))
    face = FakeLandmarker(FakeFaceResult(None))

    row = extract_frame_landmarks(pose, hand, face, _blank_frame(), frame_idx=0, ts_ms=0)

    assert row["body_nose_vis"] == 0.1
    assert row["body_nose_x"] is None
    assert "hand_right_0_x" not in row
    assert "face_mouthSmileLeft" not in row


def test_extract_frame_landmarks_no_detections_produces_minimal_row():
    pose = FakeLandmarker(FakePoseResult(None))
    hand = FakeLandmarker(FakeHandResult([]))
    face = FakeLandmarker(FakeFaceResult(None))

    row = extract_frame_landmarks(pose, hand, face, _blank_frame(), frame_idx=3, ts_ms=100)

    assert row == {"frame": 3, "timestamp_ms": 100}
