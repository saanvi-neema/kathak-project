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
