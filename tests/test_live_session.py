"""
Tests for app/live_session.py's session bookkeeping -- idle detection and
the idle-session sweep. Doesn't construct a real LiveSession (its __init__
loads real MediaPipe landmarkers and the mudra model, slow and needing real
model files) -- LiveSession.__new__ bypasses __init__ to test is_idle()/
touch() against a bare object with just the attributes they read, and
_cleanup_idle_live_sessions() is tested against lightweight stand-ins.
"""
import time

import live_session


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, last_activity, created_at, work_dir):
        self.last_activity = last_activity
        self.created_at = created_at
        self.work_dir = work_dir
        self.closed = False
        self.lock = _NullLock()

    def is_idle(self, now=None):
        now = now if now is not None else time.time()
        return (now - self.last_activity) > live_session.IDLE_TIMEOUT_SEC or \
            (now - self.created_at) > live_session.MAX_SESSION_DURATION_SEC

    def close(self):
        self.closed = True


def _bare_session():
    return live_session.LiveSession.__new__(live_session.LiveSession)


def test_is_idle_true_after_no_activity():
    now = time.time()
    session = _bare_session()
    session.last_activity = now - live_session.IDLE_TIMEOUT_SEC - 1
    session.created_at = now - live_session.IDLE_TIMEOUT_SEC - 1
    assert session.is_idle(now=now)


def test_is_idle_false_when_recently_active():
    now = time.time()
    session = _bare_session()
    session.last_activity = now
    session.created_at = now
    assert not session.is_idle(now=now)


def test_is_idle_true_past_max_session_duration_even_if_recently_active():
    now = time.time()
    session = _bare_session()
    session.last_activity = now  # just touched
    session.created_at = now - live_session.MAX_SESSION_DURATION_SEC - 1
    assert session.is_idle(now=now)


def test_touch_updates_last_activity():
    session = _bare_session()
    session.last_activity = 0
    session.touch()
    assert session.last_activity > 0


def test_cleanup_idle_live_sessions_evicts_only_idle(tmp_path, monkeypatch):
    now = time.time()
    idle_dir = tmp_path / "idle_session"
    fresh_dir = tmp_path / "fresh_session"
    idle_dir.mkdir()
    fresh_dir.mkdir()

    idle = FakeSession(now - live_session.IDLE_TIMEOUT_SEC - 10, now - live_session.IDLE_TIMEOUT_SEC - 10, str(idle_dir))
    fresh = FakeSession(now, now, str(fresh_dir))

    monkeypatch.setitem(live_session.LIVE_SESSIONS, "idle01", idle)
    monkeypatch.setitem(live_session.LIVE_SESSIONS, "fresh01", fresh)

    live_session._cleanup_idle_live_sessions()

    assert "idle01" not in live_session.LIVE_SESSIONS
    assert "fresh01" in live_session.LIVE_SESSIONS
    assert idle.closed
    assert not fresh.closed
    assert not idle_dir.exists()
    assert fresh_dir.exists()
