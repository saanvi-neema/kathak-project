"""
Dashboard backend. Serves the frontend and runs the analysis pipeline on
uploaded videos.

Usage:
    python app/server.py
    then open http://localhost:5000
"""

import dataclasses
import os
import re
import shutil
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import analyze_video, generate_pose_overlay, run_comparison
from taal_reference import TAAL_DEFINITIONS  # noqa: E402 -- scripts/ is already on sys.path via pipeline's import
from live_session import LiveSession, LIVE_SESSIONS, _cleanup_idle_live_sessions
from live_pipeline import process_live_chunk, current_snapshot, _dbg as _live_dbg
import haptic

APP_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1GB total request size -- generous for one or two dance clips, not unlimited
SESSION_MAX_AGE_HOURS = 24  # session folders (uploaded video(s) + generated overlay) older than this get swept automatically

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# DEMO MODE -- explicit, on-the-record instruction from the project owner
# (2026-09-22): the real pipeline (MediaPipe landmark extraction across
# every frame + audio beat detection) takes minutes per video, too slow for
# a live demo. Per direct request, /analyze skips the real pipeline
# entirely and returns fabricated-but-plausible numbers immediately instead
# -- the real uploaded video still plays in Overview/Pose View, only the
# scores are made up. The project owner has confirmed they will disclose
# this fabrication explicitly in their submission document. The real
# pipeline (analyze_video(), still fully intact and tested below) is one
# line away from being restored: set this back to False.
DEMO_MODE = True


def _is_allowed_video(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_VIDEO_EXTENSIONS


def _cleanup_old_sessions(max_age_hours=SESSION_MAX_AGE_HOURS):
    """
    Every /analyze and /compare call writes a permanent session folder
    (uploaded video(s) + generated overlay) that nothing ever cleaned up --
    left running a while, app/uploads/ grows without bound. A session folder
    needs to survive at least as long as the browser tab that requested it
    (it fetches the overlay/original video from /media/<session_id>/... after
    the response comes back), so this can't delete on request completion --
    instead, anything older than max_age_hours gets swept on the next request.
    """
    if not os.path.isdir(UPLOAD_DIR):
        return
    cutoff = time.time() - max_age_hours * 3600
    for name in os.listdir(UPLOAD_DIR):
        session_path = os.path.join(UPLOAD_DIR, name)
        if os.path.isdir(session_path) and os.path.getmtime(session_path) < cutoff:
            shutil.rmtree(session_path, ignore_errors=True)


@app.route("/")
def index():
    return render_template("index.html", taal_names=sorted(TAAL_DEFINITIONS))


@app.route("/v3")
def index_v3():
    return render_template("index_v3.html", taal_names=sorted(TAAL_DEFINITIONS))


def _fake_analysis_results():
    """
    DEMO_MODE's fabricated results -- see that flag's comment. Shaped
    exactly like analyze_video()'s real return dict (same keys, same
    schema per category) so the frontend can't tell the difference, but
    every number here is made up, not measured. No "landmarks_csv" key --
    the real pipeline never ran, so there's nothing for the background
    pose-overlay thread to draw from; overlay_video_filename stays None,
    which the frontend already handles as "no overlay for this video".
    """
    return {
        "duration_sec": None,
        "chakkar": {"quality_score": 94.5, "event_count": 8, "events": [], "flags": []},
        "timing": {"tempo_bpm": 92.0, "accuracy_score": 91.3, "windowed_results": [], "flags": []},
        "taal": None,
        "mudra": {
            "events": [{}] * 6,
            "rule_agreement_score": None,
            "identification_accuracy_score": 91.2,
            "flags": [],
        },
        "rasa": [
            {"start_sec": i, "end_sec": i + 1, "rasa": "shringara", "display_name": "Shringara (Love)",
             "confidence_label": "high", "mismatch_count": 0}
            for i in range(5)
        ],
        "tatkaar": {"events": [{"start_sec": 10.0, "end_sec": 12.0, "strike_count": 8, "strikes_per_sec": 4.0}] * 4},
        "overall_score": 93.4,
        "report_lines": [
            "Chakkar landings were clean and close to a full rotation throughout.",
            "Movement stayed close to the beat for most of the piece.",
            "Mudra shapes matched the expected sequence with high confidence.",
        ],
        "overlay_video_filename": None,
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    _cleanup_old_sessions()

    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    video_file = request.files["video"]
    filename = secure_filename(video_file.filename)
    if not filename:
        return jsonify({"error": "No video file selected."}), 400
    if not _is_allowed_video(filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"}), 400

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_path = os.path.join(session_dir, filename)
    video_file.save(video_path)

    # Both optional -- taal/sam aren't auto-detected (see taal_reference.py),
    # so taal analysis is just honestly skipped unless both are supplied.
    taal_name = request.form.get("taal") or None
    if taal_name is not None and taal_name not in TAAL_DEFINITIONS:
        return jsonify({"error": f"Unknown taal '{taal_name}'."}), 400
    sam_time_raw = request.form.get("sam_time")
    sam_time = None
    if sam_time_raw:
        try:
            sam_time = float(sam_time_raw)
        except ValueError:
            return jsonify({"error": "Sam time must be a number (seconds)."}), 400

    # Optional, free-text, not validated against a closed reference list --
    # unlike taal (a fixed dropdown), mudra names here have to match whatever
    # the classifier's own training labels use, so strict validation would
    # just create false rejections. Comma- or newline-separated, in the
    # order the dancer actually performed them.
    expected_mudras_raw = request.form.get("expected_mudras", "")
    expected_mudra_sequence = [
        name.strip().lower()
        for name in expected_mudras_raw.replace(",", "\n").splitlines()
        if name.strip()
    ] or None

    if DEMO_MODE:
        results = _fake_analysis_results()
    else:
        try:
            results = analyze_video(
                video_path, work_dir=session_dir, taal_name=taal_name, sam_time=sam_time,
                expected_mudra_sequence=expected_mudra_sequence,
            )
        except Exception:
            app.logger.exception("Analysis failed for session %s", session_id)
            return jsonify({"error": "Analysis failed -- check the server log for details."}), 500

        # generate_pose_overlay() alone measured ~1.2x realtime on a real clip
        # -- nearly as expensive as landmark extraction itself, and it was
        # blocking this entire response just to prepare a video for a tab
        # (Pose View) the scores callers actually want to see don't need at
        # all. Runs in the background instead; overlay_video_filename is the
        # well-known name generate_pose_overlay() always writes to, so the
        # frontend can poll for it (see app.js's pollForOverlay) instead of
        # waiting on it here.
        landmarks_csv = results.pop("landmarks_csv")
        overlay_filename = "pose_overlay.mp4"
        overlay_path = os.path.join(session_dir, overlay_filename)
        threading.Thread(
            target=generate_pose_overlay, args=(video_path, landmarks_csv, overlay_path), daemon=True,
        ).start()
        results["overlay_video_filename"] = overlay_filename

    results["session_id"] = session_id
    results["original_video_filename"] = filename
    return jsonify(serialize_results(results))


@app.route("/compare", methods=["POST"])
def compare():
    _cleanup_old_sessions()

    if "teacher_video" not in request.files or "student_video" not in request.files:
        return jsonify({"error": "Both a teacher video and a student video are required."}), 400

    teacher_file = request.files["teacher_video"]
    student_file = request.files["student_video"]
    if not teacher_file.filename or not student_file.filename:
        return jsonify({"error": "Both a teacher video and a student video are required."}), 400

    teacher_filename = secure_filename(teacher_file.filename)
    student_filename = secure_filename(student_file.filename)
    if not teacher_filename or not student_filename:
        return jsonify({"error": "Both a teacher video and a student video are required."}), 400
    if not _is_allowed_video(teacher_filename) or not _is_allowed_video(student_filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"}), 400

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    teacher_path = os.path.join(session_dir, "teacher_" + teacher_filename)
    student_path = os.path.join(session_dir, "student_" + student_filename)
    teacher_file.save(teacher_path)
    student_file.save(student_path)

    try:
        results = run_comparison(teacher_path, student_path, work_dir=session_dir)
    except Exception:
        app.logger.exception("Comparison failed for session %s", session_id)
        return jsonify({"error": "Comparison failed -- check the server log for details."}), 500

    if results is None:
        return jsonify({"error": "Couldn't compare these videos -- one or both didn't have "
                                  "enough audio signal (claps/stomps/music) to align and compare."}), 422

    results["session_id"] = session_id
    return jsonify(serialize_results(results))


@app.route("/live/start", methods=["POST"])
def live_start():
    """
    Allocates a live capture session -- mirrors /analyze's optional
    taal/sam_time/expected_mudras fields and uuid session-id pattern, but
    keeps state in LIVE_SESSIONS (persistent landmarkers, rolling buffers)
    instead of writing an uploaded file. See app/live_session.py.
    """
    _cleanup_old_sessions()
    _cleanup_idle_live_sessions()

    taal_name = request.form.get("taal") or None
    if taal_name is not None and taal_name not in TAAL_DEFINITIONS:
        return jsonify({"error": f"Unknown taal '{taal_name}'."}), 400
    sam_time_raw = request.form.get("sam_time")
    sam_time = None
    if sam_time_raw:
        try:
            sam_time = float(sam_time_raw)
        except ValueError:
            return jsonify({"error": "Sam time must be a number (seconds)."}), 400

    expected_mudras_raw = request.form.get("expected_mudras", "")
    expected_mudra_sequence = [
        name.strip().lower()
        for name in expected_mudras_raw.replace(",", "\n").splitlines()
        if name.strip()
    ] or None

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    session = LiveSession(
        session_id, session_dir, taal_name=taal_name, sam_time=sam_time,
        expected_mudra_sequence=expected_mudra_sequence,
    )
    LIVE_SESSIONS[session_id] = session
    return jsonify({"session_id": session_id})


@app.route("/live/chunk", methods=["POST"])
def live_chunk():
    """
    Accepts one recorded chunk (raw bytes in the request body, same as the
    live-capture spike that validated this approach) and returns a full
    results snapshot -- see live_pipeline.process_live_chunk.
    """
    _cleanup_idle_live_sessions()

    session_id = request.args.get("session_id", "")
    if not SESSION_ID_RE.match(session_id):
        return jsonify({"error": "Invalid session id."}), 404
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        return jsonify({"error": "Unknown or expired live session."}), 404

    ext = request.args.get("ext", "webm")
    if ext not in {"webm", "mp4"}:
        return jsonify({"error": "Unsupported chunk container."}), 400
    try:
        duration_sec = float(request.args.get("duration_sec"))
    except (TypeError, ValueError):
        return jsonify({"error": "duration_sec is required and must be a number."}), 400
    # The browser cycles ~1.5s chunks (see app.js's LIVE_CHUNK_MS) -- a wildly
    # implausible value (clock drift on a throttled/backgrounded mobile tab,
    # or a buggy client) would silently corrupt live_pipeline's effective-fps
    # math and the finalize-margin invariant it depends on, rather than fail
    # loudly. 10s is a generous ceiling, not the expected value.
    if not (0 < duration_sec <= 10.0):
        return jsonify({"error": "duration_sec out of expected range (0, 10] seconds."}), 400

    chunk_bytes = request.get_data()
    _live_dbg(f"chunk={len(chunk_bytes)}b ext={ext} duration={duration_sec:.2f}s")
    if not chunk_bytes:
        return jsonify({"error": "No chunk data received."}), 400

    with session.lock:
        # Re-check the session is still live now that the lock is held --
        # /live/stop or the idle sweep could have popped and closed it (its
        # own close() + rmtree also run under this same lock) while this
        # request was queued waiting for the lock, in which case `session`
        # here is a stale reference to already-closed landmarkers/a deleted
        # work_dir. Only matters under real request concurrency (the current
        # dev server is single-threaded), but cheap to guard regardless.
        if LIVE_SESSIONS.get(session_id) is not session:
            return jsonify({"error": "Unknown or expired live session."}), 404
        try:
            snapshot = process_live_chunk(session, chunk_bytes, ext, duration_sec)
        except Exception:
            app.logger.exception("Live chunk processing failed for session %s", session_id)
            return jsonify({"error": "Live chunk processing failed -- check the server log for details."}), 500
        session.touch()

    snapshot["session_id"] = session_id
    return jsonify(serialize_results(snapshot))


@app.route("/live/stop", methods=["POST"])
def live_stop():
    session_id = request.args.get("session_id", "") or request.form.get("session_id", "")
    if not SESSION_ID_RE.match(session_id):
        return jsonify({"error": "Invalid session id."}), 404
    session = LIVE_SESSIONS.pop(session_id, None)
    if session is None:
        return jsonify({"error": "Unknown or expired live session."}), 404

    with session.lock:
        snapshot = current_snapshot(session)
        session.close()
    shutil.rmtree(session.work_dir, ignore_errors=True)

    snapshot["session_id"] = session_id
    return jsonify(serialize_results(snapshot))


SESSION_ID_RE = re.compile(r"^[0-9a-f]{12}$")  # matches uuid.uuid4().hex[:12], the only format /analyze and /compare ever generate


@app.route("/media/<session_id>/<filename>")
def media(session_id, filename):
    """
    Real bug found and fixed in an audit: session_id came straight from the
    URL into send_from_directory()'s DIRECTORY argument (not just the
    filename) -- Flask's own docs call this out explicitly as unsafe,
    since send_from_directory only sanitizes the filename against path
    traversal, not the directory. Confirmed directly, not just theorized:
    a request to /media/../server.py returned this file's own source before
    this fix. Every session_id this app ever generates is
    uuid.uuid4().hex[:12] -- a fixed-length lowercase hex string -- so
    anything that doesn't match that shape is rejected outright, closing
    off "..", "/", and anything else that isn't a real session id.
    """
    if not SESSION_ID_RE.match(session_id):
        return jsonify({"error": "Invalid session id."}), 404
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    return send_from_directory(session_dir, filename)


def serialize_results(results):
    """Convert Flag dataclasses, numpy types, etc. into plain JSON-safe values."""
    def clean(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return clean(dataclasses.asdict(v))
        if isinstance(v, dict):
            return {k: clean(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if hasattr(v, "item"):  # numpy scalar
            return v.item()
        return v
    return clean(results)


@app.route("/test/haptic/<int:finger>")
def test_haptic(finger):
    """Test route: buzz one motor by finger index (0=thumb … 4=pinky)."""
    ok = haptic.buzz_finger(finger)
    return jsonify({"finger": finger, "sent": ok})


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _cleanup_old_sessions()
    app.run(debug=True, port=5000, use_reloader=False)
