"""
Live capture session state.

A live session needs a few things kept alive across many short chunk
uploads instead of recreated per chunk: the three MediaPipe landmarkers
(reloading their .task model bundles every ~1-2 seconds would add real
per-chunk latency) and the mudra classifier's model bundle (the real file
is 174MB -- reloading it from disk every chunk would likely cost more than
the MediaPipe inference itself). This module holds that state and the
rolling landmarks/audio buffers a live session accumulates.

See the live-capture plan for the overall design; app/live_pipeline.py is
what actually processes chunks against a LiveSession.
"""
import os
import shutil
import sys
import threading
import time

import numpy as np

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from extract_landmarks import make_pose_landmarker, make_hand_landmarker, make_face_landmarker  # noqa: E402
from pipeline import MUDRA_MODEL_PATH  # noqa: E402

AUDIO_SR = 22050  # matches extract_audio()'s fixed ffmpeg resample target (-ar 22050 -ac 1) so per-chunk audio concatenates cleanly
IDLE_TIMEOUT_SEC = 5 * 60  # no chunk uploaded in this long -> assume the tab closed without a clean /live/stop
MAX_SESSION_DURATION_SEC = 30 * 60  # hard backstop regardless of activity, so a forgotten tab can't grow buffers forever
TIMING_RECOMPUTE_INTERVAL_SEC = 4.0  # how often to re-run the audio beat-grid + timing/taal checks -- the only checks expensive enough (librosa tempo detection over the whole rolling buffer) to need throttling


class LiveSession:
    def __init__(self, session_id, work_dir, taal_name=None, sam_time=None, expected_mudra_sequence=None):
        self.session_id = session_id
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        self.taal_name = taal_name
        self.sam_time = sam_time
        self.expected_mudra_sequence = expected_mudra_sequence

        self.pose_landmarker = make_pose_landmarker()
        self.hand_landmarker = make_hand_landmarker()
        self.face_landmarker = make_face_landmarker()

        self.model_bundle = None
        if os.path.exists(MUDRA_MODEL_PATH):
            import joblib
            self.model_bundle = joblib.load(MUDRA_MODEL_PATH)

        # Continuing frame/timestamp counters -- next_video_timestamp_ms
        # (scripts/extract_landmarks.py) needs these to keep advancing
        # seamlessly across chunk boundaries instead of resetting to 0.
        self.frame_idx = 0
        self.last_ts = -1
        self.effective_fps = 30.0  # updated from each chunk's actual decoded frame count -- cv2's reported fps/frame-count metadata is unreliable for these short recorder-cycled chunks (confirmed directly against real phone-recorded chunks before building this)

        self.landmarks_rows = []  # schema matches extract_landmarks.py's CSV rows
        self.duration_sec = 0.0  # total logical video duration accumulated so far

        self.audio_samples = np.zeros(0, dtype=np.float32)
        self.sr = AUDIO_SR

        # What's already been reported to the client. Chakkar is
        # append-once (a finalized event never changes). Rasa/mudra are
        # keyed dicts (start_sec, or (hand_side, start_sec) for mudra) --
        # a window/hold's underlying frames are fixed once they're in the
        # past, so a given key's value is stable once finalized, but a
        # position-based list-append isn't safe for rasa: a window with no
        # detected face is skipped entirely by run_rasa_analysis, which
        # would otherwise shift list positions across ticks.
        self.chakkar_events = []
        self.rasa_events = {}
        self.mudra_events = {}
        self.timing = None
        self.taal = None

        self.cached_beat_grid = None
        self.last_timing_recompute_wall = 0.0

        self.created_at = time.time()
        self.last_activity = time.time()
        self.lock = threading.Lock()

    @property
    def landmarks_csv_path(self):
        return os.path.join(self.work_dir, "live_landmarks.csv")

    def touch(self):
        self.last_activity = time.time()

    def is_idle(self, now=None):
        now = now if now is not None else time.time()
        return (now - self.last_activity) > IDLE_TIMEOUT_SEC or (now - self.created_at) > MAX_SESSION_DURATION_SEC

    def close(self):
        self.pose_landmarker.close()
        self.hand_landmarker.close()
        self.face_landmarker.close()


LIVE_SESSIONS = {}


def _cleanup_idle_live_sessions():
    """
    Mirrors app/server.py's _cleanup_old_sessions pattern, but for
    in-memory live sessions rather than on-disk upload folders -- a tab
    closed without a clean /live/stop (network drop, browser crash) would
    otherwise leak a LiveSession's landmarkers (native resources) forever.
    """
    now = time.time()
    stale_ids = [sid for sid, session in LIVE_SESSIONS.items() if session.is_idle(now)]
    for sid in stale_ids:
        session = LIVE_SESSIONS.pop(sid)
        session.close()
        shutil.rmtree(session.work_dir, ignore_errors=True)
