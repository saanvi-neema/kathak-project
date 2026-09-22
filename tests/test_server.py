"""
Tests for the housekeeping pieces added to app/server.py: file-type
validation and automatic cleanup of old session folders. Most of this
module's logic is thin glue around analyze_video()/run_comparison(),
already covered indirectly through their own module tests -- but the
/media route is tested directly (via Flask's test client) below, because
"thin glue" is exactly where a real path-traversal bug was found (see
test_media_route_rejects_path_traversal_attempts). Route-level behavior
isn't automatically safe just because the underlying logic is simple.
"""

import io
import os
import time

import server


def test_media_route_rejects_path_traversal_attempts():
    """
    Real bug found in an audit, confirmed by actually exploiting it before
    the fix: session_id came straight from the URL into
    send_from_directory()'s DIRECTORY argument, which Flask's own docs warn
    is unsafe (only the filename argument is sanitized against traversal).
    A request to /media/../server.py returned this project's own server
    source before the fix. Tests the exact attack, plus URL-encoded
    variants, and confirms a well-formed-but-nonexistent session id still
    gets an ordinary 404 (not wrongly rejected as invalid).
    """
    server.app.testing = True
    client = server.app.test_client()

    for path in ["/media/../server.py", "/media/..%2Fserver.py", "/media/%2e%2e/server.py",
                 "/media/not-a-hex-session-id/server.py"]:
        resp = client.get(path)
        assert resp.status_code == 404
        assert resp.get_json() == {"error": "Invalid session id."}, \
            f"{path} should be rejected by the session_id format check, not merely 404 from a missing file"

    resp = client.get("/media/0123456789ab/foo.mp4")
    assert resp.status_code == 404
    assert resp.get_json() != {"error": "Invalid session id."}, \
        "a well-formed session id (even if the session doesn't exist) should reach send_from_directory, not the format check"


def test_allowed_video_extensions_accepted():
    for name in ["clip.mp4", "CLIP.MOV", "video.mkv", "x.webm", "y.avi", "z.m4v"]:
        assert server._is_allowed_video(name), f"{name} should be allowed"


def test_disallowed_extensions_rejected():
    for name in ["notes.txt", "script.py", "archive.zip", "no_extension", "video.exe"]:
        assert not server._is_allowed_video(name), f"{name} should be rejected"


def test_cleanup_removes_only_old_session_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path))

    old_session = tmp_path / "old_session"
    new_session = tmp_path / "new_session"
    old_session.mkdir()
    new_session.mkdir()
    (old_session / "video.mp4").write_bytes(b"x")
    (new_session / "video.mp4").write_bytes(b"x")

    old_time = time.time() - 48 * 3600  # 48 hours ago
    os.utime(old_session, (old_time, old_time))

    server._cleanup_old_sessions(max_age_hours=24)

    assert not old_session.exists(), "session older than max_age_hours should be removed"
    assert new_session.exists(), "recent session should be left alone"


def test_cleanup_handles_missing_upload_dir(tmp_path, monkeypatch):
    missing_dir = tmp_path / "does_not_exist"
    monkeypatch.setattr(server, "UPLOAD_DIR", str(missing_dir))
    server._cleanup_old_sessions()  # should not raise


def test_analyze_route_backgrounds_overlay_generation_and_returns_json(tmp_path, monkeypatch):
    """
    Real bug found and fixed: generate_pose_overlay() was moved out of
    analyze_video() and into a background thread here in /analyze (see
    pipeline.py's analyze_video docstring -- it measured ~1.2x realtime,
    nearly as expensive as landmark extraction, purely to prepare a video
    for a tab most callers aren't looking at yet). That threading.Thread(..)
    call, and the analyze_video()-returned landmarks_csv it depends on,
    were only ever exercised by actually running the full multi-minute
    pipeline on a real video -- nothing in this suite does that, so a
    missing `import threading` here went uncaught: every real upload 500'd
    with NameError, and because that happened outside analyze()'s own
    try/except, Flask's default handler returned an HTML error page
    instead of this project's usual jsonify()'d error -- which is why the
    browser reported "Unexpected token '<' ... is not valid JSON" instead
    of a real error message. Mocking analyze_video/generate_pose_overlay
    exercises this route's actual code path in milliseconds instead.
    """
    # DEMO_MODE (see its own comment in server.py) bypasses analyze_video()
    # entirely -- forced off here since this test exists specifically to
    # cover that real code path, independent of whatever DEMO_MODE currently
    # defaults to.
    monkeypatch.setattr(server, "DEMO_MODE", False)
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path))

    fake_results = {
        "duration_sec": 1.0, "chakkar": None, "timing": None, "taal": None,
        "mudra": None, "rasa": None, "tatkaar": None, "overall_score": None,
        "report_lines": [], "landmarks_csv": str(tmp_path / "landmarks.csv"),
    }
    monkeypatch.setattr(server, "analyze_video", lambda *a, **kw: dict(fake_results))

    overlay_calls = []
    monkeypatch.setattr(server, "generate_pose_overlay", lambda *a, **kw: overlay_calls.append(a))

    server.app.testing = True
    client = server.app.test_client()
    resp = client.post(
        "/analyze",
        data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["overlay_video_filename"] == "pose_overlay.mp4"
    assert "session_id" in data
    assert data["original_video_filename"] == "clip.mp4"
    assert "landmarks_csv" not in data, "internal file path must not leak into the JSON response"

    for _ in range(40):
        if overlay_calls:
            break
        time.sleep(0.05)
    assert overlay_calls, "generate_pose_overlay should have been scheduled in a background thread"


def test_demo_mode_returns_fake_results_without_running_the_real_pipeline(tmp_path, monkeypatch):
    """
    DEMO_MODE (see server.py's comment on the flag -- explicit, disclosed
    project-owner instruction) must never touch analyze_video() or
    generate_pose_overlay(): the whole point is skipping the real,
    multi-minute pipeline. Fails loudly if DEMO_MODE's fake path ever
    starts calling into the real one by accident.
    """
    monkeypatch.setattr(server, "DEMO_MODE", True)
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path))

    def _must_not_be_called(*a, **kw):
        raise AssertionError("DEMO_MODE must not invoke the real pipeline")

    monkeypatch.setattr(server, "analyze_video", _must_not_be_called)
    monkeypatch.setattr(server, "generate_pose_overlay", _must_not_be_called)

    server.app.testing = True
    client = server.app.test_client()
    resp = client.post(
        "/analyze",
        data={"video": (io.BytesIO(b"fake video bytes"), "clip.mp4")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["overall_score"] is not None
    assert data["overlay_video_filename"] is None
    assert data["original_video_filename"] == "clip.mp4"
    assert "landmarks_csv" not in data
