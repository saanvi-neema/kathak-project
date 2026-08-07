"""
Flask test-client tests for the live-capture routes (/live/start,
/live/chunk, /live/stop) -- request validation and session lookup, same
"thin glue tested directly" philosophy as test_server.py's existing route
tests. LiveSession construction and process_live_chunk/current_snapshot are
monkeypatched away: real LiveSession.__init__ loads MediaPipe landmarkers
and the mudra model (slow, needs real model files), and that correctness
is covered separately (test_live_pipeline.py's pure-logic tests, plus the
live-capture plan's manual verification against real recorded chunks).
"""
import pytest

import live_session
import server


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeLiveSession:
    def __init__(self, session_id, work_dir, taal_name=None, sam_time=None, expected_mudra_sequence=None):
        self.session_id = session_id
        self.work_dir = work_dir
        self.taal_name = taal_name
        self.sam_time = sam_time
        self.expected_mudra_sequence = expected_mudra_sequence
        self.lock = _NullLock()
        self.closed = False
        self.last_activity = None

    def close(self):
        self.closed = True

    def touch(self):
        self.last_activity = "touched"

    def is_idle(self, now=None):
        # _cleanup_idle_live_sessions() runs for real on every /live/start
        # and /live/chunk call -- this fake never reports idle, so it isn't
        # unexpectedly evicted mid-test.
        return False


FAKE_SNAPSHOT = {
    "duration_sec": 0.0, "chakkar": None, "timing": None, "taal": None,
    "mudra": None, "rasa": None, "overall_score": None, "report_lines": [],
    "overlay_video_filename": None, "original_video_filename": None,
}


@pytest.fixture(autouse=True)
def _clean_live_sessions():
    live_session.LIVE_SESSIONS.clear()
    yield
    live_session.LIVE_SESSIONS.clear()


def _client():
    server.app.testing = True
    return server.app.test_client()


def test_live_start_creates_a_session(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    resp = _client().post("/live/start")
    assert resp.status_code == 200
    session_id = resp.get_json()["session_id"]
    assert session_id in server.LIVE_SESSIONS
    assert isinstance(server.LIVE_SESSIONS[session_id], FakeLiveSession)


def test_live_start_rejects_unknown_taal(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    resp = _client().post("/live/start", data={"taal": "not-a-real-taal"})
    assert resp.status_code == 400


def test_live_start_rejects_non_numeric_sam_time(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    resp = _client().post("/live/start", data={"sam_time": "not-a-number"})
    assert resp.status_code == 400


def test_live_start_parses_expected_mudras(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    resp = _client().post("/live/start", data={"expected_mudras": "Pataka, Tripataka\nMushti"})
    session_id = resp.get_json()["session_id"]
    session = server.LIVE_SESSIONS[session_id]
    assert session.expected_mudra_sequence == ["pataka", "tripataka", "mushti"]


def test_live_chunk_rejects_invalid_session_id():
    resp = _client().post("/live/chunk?session_id=not-hex&ext=webm&duration_sec=1.5", data=b"x")
    assert resp.status_code == 404
    assert resp.get_json() == {"error": "Invalid session id."}


def test_live_chunk_rejects_unknown_session():
    resp = _client().post("/live/chunk?session_id=0123456789ab&ext=webm&duration_sec=1.5", data=b"x")
    assert resp.status_code == 404


def test_live_chunk_missing_duration_sec_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    client = _client()
    session_id = client.post("/live/start").get_json()["session_id"]
    resp = client.post(f"/live/chunk?session_id={session_id}&ext=webm", data=b"x")
    assert resp.status_code == 400


def test_live_chunk_unsupported_ext_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    client = _client()
    session_id = client.post("/live/start").get_json()["session_id"]
    resp = client.post(f"/live/chunk?session_id={session_id}&ext=avi&duration_sec=1.5", data=b"x")
    assert resp.status_code == 400


def test_live_chunk_empty_body_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    client = _client()
    session_id = client.post("/live/start").get_json()["session_id"]
    resp = client.post(f"/live/chunk?session_id={session_id}&ext=webm&duration_sec=1.5", data=b"")
    assert resp.status_code == 400


def test_live_chunk_processes_against_known_session(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    client = _client()
    session_id = client.post("/live/start").get_json()["session_id"]

    called = {}

    def fake_process(session, chunk_bytes, ext, duration_sec):
        called["args"] = (chunk_bytes, ext, duration_sec)
        return dict(FAKE_SNAPSHOT)

    monkeypatch.setattr(server, "process_live_chunk", fake_process)

    resp = client.post(f"/live/chunk?session_id={session_id}&ext=webm&duration_sec=1.5", data=b"chunkdata")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["session_id"] == session_id
    assert called["args"] == (b"chunkdata", "webm", 1.5)
    assert server.LIVE_SESSIONS[session_id].last_activity == "touched"


def test_live_chunk_processing_failure_returns_500_not_a_crash(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    client = _client()
    session_id = client.post("/live/start").get_json()["session_id"]

    def failing_process(session, chunk_bytes, ext, duration_sec):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "process_live_chunk", failing_process)

    resp = client.post(f"/live/chunk?session_id={session_id}&ext=webm&duration_sec=1.5", data=b"x")
    assert resp.status_code == 500
    # the session should still exist -- one bad chunk doesn't tear down the session
    assert session_id in server.LIVE_SESSIONS


def test_live_stop_evicts_session_and_returns_snapshot(monkeypatch):
    monkeypatch.setattr(server, "LiveSession", FakeLiveSession)
    monkeypatch.setattr(server, "current_snapshot", lambda session: dict(FAKE_SNAPSHOT))
    client = _client()
    session_id = client.post("/live/start").get_json()["session_id"]
    fake_session = server.LIVE_SESSIONS[session_id]

    resp = client.post(f"/live/stop?session_id={session_id}")
    assert resp.status_code == 200
    assert resp.get_json()["session_id"] == session_id
    assert session_id not in server.LIVE_SESSIONS
    assert fake_session.closed

    # a chunk after stop should 404, not resurrect the session
    resp2 = client.post(f"/live/chunk?session_id={session_id}&ext=webm&duration_sec=1.5", data=b"x")
    assert resp2.status_code == 404


def test_live_stop_unknown_session_returns_404():
    resp = _client().post("/live/stop?session_id=0123456789ab")
    assert resp.status_code == 404
