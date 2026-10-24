import logging
import time
import uuid
from pathlib import Path

import cv2
from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from database import db
from detection.image_processor import process_image
from detection.plate_detector import ModelUnavailableError
from detection.video_processor import process_video

api = Blueprint("api", __name__, url_prefix="/api")
logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "m4v"}


def _error(message, status):
    return jsonify({"ok": False, "error": message}), status


def _save_upload(file, kind):
    if not file or not file.filename:
        raise ValueError("Choose a file to upload.")
    filename = secure_filename(file.filename)
    if not filename or "." not in filename:
        raise ValueError("The uploaded filename is invalid.")
    extension = filename.rsplit(".", 1)[1].lower()
    allowed = IMAGE_EXTENSIONS if kind == "image" else VIDEO_EXTENSIONS
    if extension not in allowed:
        raise ValueError(f"Unsupported {kind} format. Allowed: {', '.join(sorted(allowed))}.")
    target_dir = current_app.config["UPLOAD_IMAGE_DIR"] if kind == "image" else current_app.config["UPLOAD_VIDEO_DIR"]
    path = Path(target_dir) / f"{uuid.uuid4().hex}.{extension}"
    file.save(path)
    if kind == "image":
        image = cv2.imread(str(path))
        if image is None:
            path.unlink(missing_ok=True)
            raise ValueError("The uploaded file is not a readable image.")
    else:
        capture = cv2.VideoCapture(str(path))
        valid = capture.isOpened()
        capture.release()
        if not valid:
            path.unlink(missing_ok=True)
            raise ValueError("The uploaded file is not a readable video.")
    return filename, path


def _within_managed(path, settings):
    resolved = Path(path).resolve()
    roots = [Path(settings.ANNOTATED_DIR), Path(settings.CROPS_DIR), Path(settings.VIDEO_OUTPUT_DIR),
             Path(settings.UPLOAD_IMAGE_DIR), Path(settings.UPLOAD_VIDEO_DIR)]
    return any(resolved.is_relative_to(root.resolve()) for root in roots)


@api.get("/health")
def health():
    model_path = Path(current_app.config["MODEL_PATH"])
    return jsonify({"ok": True, "status": "ready", "plate_model_available": model_path.is_file()})


@api.get("/stats")
def stats():
    return jsonify(db.get_stats(current_app.config["DATABASE_PATH"]))


@api.get("/analytics")
def analytics():
    return jsonify(db.get_analytics(current_app.config["DATABASE_PATH"]))


@api.route("/history", methods=["GET"])
def history():
    page = request.args.get("page", 1, type=int)
    if page < 1:
        return _error("page must be positive", 400)
    jobs, total = db.list_jobs(current_app.config["DATABASE_PATH"], request.args.get("search", ""),
                               request.args.get("type", ""), request.args.get("date", ""), page)
    return jsonify({"items": jobs, "total": total, "page": page, "per_page": 20})


@api.post("/detect/image")
def detect_image():
    try:
        original, path = _save_upload(request.files.get("file"), "image")
    except ValueError as exc:
        return _error(str(exc), 400)
    job_id = db.create_job(current_app.config["DATABASE_PATH"], "image", original, path)
    started = time.perf_counter()
    try:
        detections, output, elapsed = process_image(path, job_id, current_app.config["SETTINGS"])
        for detection in detections:
            detection_id = db.add_detection(current_app.config["DATABASE_PATH"], job_id, detection)
            detection["crop_url"] = f"/api/file/{job_id}/crop/{detection_id}"
        db.finish_job(current_app.config["DATABASE_PATH"], job_id, "completed", elapsed, output)
        return jsonify({"ok": True, "job_id": job_id, "detections": detections, "annotated_url": f"/files/{job_id}/annotated",
                        "processing_seconds": elapsed}), 200
    except ModelUnavailableError as exc:
        db.finish_job(current_app.config["DATABASE_PATH"], job_id, "failed", time.perf_counter() - started, error_message=str(exc))
        return _error(str(exc), 503)
    except Exception as exc:
        logger.exception("Image processing failed for job %s", job_id)
        db.finish_job(current_app.config["DATABASE_PATH"], job_id, "failed", time.perf_counter() - started, error_message=str(exc))
        return _error("Image processing failed. Check the server log for details.", 500)


@api.post("/detect/video")
def detect_video():
    try:
        original, path = _save_upload(request.files.get("file"), "video")
    except ValueError as exc:
        return _error(str(exc), 400)
    job_id = db.create_job(current_app.config["DATABASE_PATH"], "video", original, path)
    started = time.perf_counter()
    try:
        detections, output, elapsed = process_video(path, job_id, current_app.config["SETTINGS"])
        for detection in detections:
            detection_id = db.add_detection(current_app.config["DATABASE_PATH"], job_id, detection)
            detection["crop_url"] = f"/api/file/{job_id}/crop/{detection_id}"
        db.finish_job(current_app.config["DATABASE_PATH"], job_id, "completed", elapsed, output)
        return jsonify({"ok": True, "job_id": job_id, "detections": detections,
                        "output_url": f"/files/{job_id}/video", "processing_seconds": elapsed}), 200
    except ModelUnavailableError as exc:
        db.finish_job(current_app.config["DATABASE_PATH"], job_id, "failed", time.perf_counter() - started, error_message=str(exc))
        return _error(str(exc), 503)
    except Exception as exc:
        logger.exception("Video processing failed for job %s", job_id)
        db.finish_job(current_app.config["DATABASE_PATH"], job_id, "failed", time.perf_counter() - started, error_message=str(exc))
        return _error("Video processing failed. Check the server log for details.", 500)


@api.get("/history/<int:job_id>")
def history_detail(job_id):
    item = db.get_job(current_app.config["DATABASE_PATH"], job_id)
    return (jsonify(item), 200) if item else _error("History record not found.", 404)


@api.delete("/history/<int:job_id>")
def delete_history(job_id):
    item = db.get_job(current_app.config["DATABASE_PATH"], job_id)
    if not item:
        return _error("History record not found.", 404)
    settings = current_app.config["SETTINGS"]
    paths = [item.get("input_path"), item.get("output_path"), *(d.get("crop_path") for d in item["detections"])]
    if not all(not path or _within_managed(path, settings) for path in paths):
        return _error("Record contains a path outside managed storage; files were not removed.", 409)
    db.delete_job(current_app.config["DATABASE_PATH"], job_id)
    for path in paths:
        if path:
            Path(path).unlink(missing_ok=True)
    return jsonify({"ok": True})


@api.get("/file/<int:job_id>/<kind>/<int:detection_id>")
def detection_file(job_id, kind, detection_id):
    job = db.get_job(current_app.config["DATABASE_PATH"], job_id)
    if not job:
        return _error("History record not found.", 404)
    detection = next((d for d in job["detections"] if d["id"] == detection_id), None)
    if kind == "crop" and detection:
        path = detection["crop_path"]
    elif kind == "annotated":
        path = job.get("output_path") if job["job_type"] == "image" else None
    elif kind == "video":
        path = job.get("output_path") if job["job_type"] == "video" else None
    else:
        path = None
    if not path or not _within_managed(path, current_app.config["SETTINGS"]) or not Path(path).is_file():
        return _error("Managed output not found.", 404)
    return send_file(path, as_attachment=kind in ("crop", "annotated", "video"))
