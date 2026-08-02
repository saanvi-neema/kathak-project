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
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import analyze_video, run_comparison
from taal_reference import TAAL_DEFINITIONS  # noqa: E402 -- scripts/ is already on sys.path via pipeline's import

APP_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1GB total request size -- generous for one or two dance clips, not unlimited
SESSION_MAX_AGE_HOURS = 24  # session folders (uploaded video(s) + generated overlay) older than this get swept automatically

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


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

    try:
        results = analyze_video(
            video_path, work_dir=session_dir, taal_name=taal_name, sam_time=sam_time,
            expected_mudra_sequence=expected_mudra_sequence,
        )
    except Exception:
        app.logger.exception("Analysis failed for session %s", session_id)
        return jsonify({"error": "Analysis failed -- check the server log for details."}), 500

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


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _cleanup_old_sessions()
    app.run(debug=True, port=5000)
