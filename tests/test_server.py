"""
Tests for the housekeeping pieces added to app/server.py: file-type
validation and automatic cleanup of old session folders. Only these pure
helpers are tested, not the Flask routes themselves -- routing/upload
handling is thin glue around analyze_video()/run_comparison(), which are
already covered indirectly through their own module tests.
"""

import os
import time

import server


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
