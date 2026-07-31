"""
Dashboard backend. Serves the frontend and runs the analysis pipeline on
uploaded videos.

Usage:
    python app/server.py
    then open http://localhost:5000
"""

import dataclasses
import os
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from pipeline import analyze_video, run_comparison

APP_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded."}), 400

    video_file = request.files["video"]
    filename = secure_filename(video_file.filename)
    if not filename:
        return jsonify({"error": "No video file selected."}), 400

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_path = os.path.join(session_dir, filename)
    video_file.save(video_path)

    try:
        results = analyze_video(video_path, work_dir=session_dir)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    results["session_id"] = session_id
    results["original_video_filename"] = filename
    return jsonify(serialize_results(results))


@app.route("/compare", methods=["POST"])
def compare():
    if "teacher_video" not in request.files or "student_video" not in request.files:
        return jsonify({"error": "Both a teacher video and a student video are required."}), 400

    teacher_file = request.files["teacher_video"]
    student_file = request.files["student_video"]
    if not teacher_file.filename or not student_file.filename:
        return jsonify({"error": "Both a teacher video and a student video are required."}), 400

    session_id = uuid.uuid4().hex[:12]
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    teacher_path = os.path.join(session_dir, "teacher_" + teacher_file.filename)
    student_path = os.path.join(session_dir, "student_" + student_file.filename)
    teacher_file.save(teacher_path)
    student_file.save(student_path)

    try:
        results = run_comparison(teacher_path, student_path, work_dir=session_dir)
    except Exception as e:
        return jsonify({"error": f"Comparison failed: {e}"}), 500

    if results is None:
        return jsonify({"error": "Couldn't compare these videos -- one or both didn't have "
                                  "enough audio signal (claps/stomps/music) to align and compare."}), 422

    results["session_id"] = session_id
    return jsonify(serialize_results(results))


@app.route("/media/<session_id>/<filename>")
def media(session_id, filename):
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
    app.run(debug=True, port=5000)
