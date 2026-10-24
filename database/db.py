import sqlite3
from pathlib import Path


def connect(database_path):
    connection = sqlite3.connect(str(database_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path):
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    with connect(path) as connection:
        connection.executescript(schema)


def create_job(database_path, job_type, original_filename, input_path):
    with connect(database_path) as connection:
        cursor = connection.execute(
            "INSERT INTO jobs(job_type, original_filename, input_path, status) VALUES (?, ?, ?, 'processing')",
            (job_type, original_filename, str(input_path)),
        )
        return cursor.lastrowid


def finish_job(database_path, job_id, status, processing_seconds, output_path=None, error_message=None):
    with connect(database_path) as connection:
        connection.execute(
            "UPDATE jobs SET status=?, processing_seconds=?, output_path=?, error_message=? WHERE id=?",
            (status, processing_seconds, str(output_path) if output_path else None, error_message, job_id),
        )


def add_detection(database_path, job_id, detection):
    with connect(database_path) as connection:
        cursor = connection.execute(
            """INSERT INTO detections(job_id, plate_text, raw_ocr_text, detection_confidence,
               ocr_confidence, crop_path, frame_number, timestamp_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, detection.get("plate_text"), detection.get("raw_ocr_text"),
             detection["detection_confidence"], detection.get("ocr_confidence"),
             str(detection["crop_path"]), detection.get("frame_number"), detection.get("timestamp_seconds")),
        )
        return cursor.lastrowid


def get_job(database_path, job_id):
    with connect(database_path) as connection:
        job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not job:
            return None
        result = dict(job)
        result["detections"] = [dict(row) for row in connection.execute(
            "SELECT * FROM detections WHERE job_id=? ORDER BY id", (job_id,)
        )]
        return result


def list_jobs(database_path, search="", job_type="", date="", page=1, per_page=20):
    clauses, values = [], []
    if search:
        clauses.append("EXISTS (SELECT 1 FROM detections d WHERE d.job_id=jobs.id AND (d.plate_text LIKE ? OR d.raw_ocr_text LIKE ?))")
        values.extend((f"%{search}%", f"%{search}%"))
    if job_type in ("image", "video"):
        clauses.append("job_type=?")
        values.append(job_type)
    if date:
        clauses.append("date(created_at)=?")
        values.append(date)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    offset = (max(1, page) - 1) * per_page
    with connect(database_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM jobs" + where, values).fetchone()[0]
        rows = connection.execute("SELECT * FROM jobs" + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
                                  [*values, per_page, offset]).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["detections"] = [dict(d) for d in connection.execute(
                "SELECT * FROM detections WHERE job_id=? ORDER BY id", (item["id"],)
            )]
            items.append(item)
        return items, total


def delete_job(database_path, job_id):
    with connect(database_path) as connection:
        cursor = connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return cursor.rowcount > 0


def get_stats(database_path):
    with connect(database_path) as connection:
        return dict(connection.execute("""SELECT
            COALESCE(SUM(CASE WHEN job_type='image' AND status='completed' THEN 1 ELSE 0 END), 0) AS images_processed,
            COALESCE(SUM(CASE WHEN job_type='video' AND status='completed' THEN 1 ELSE 0 END), 0) AS videos_processed,
            COALESCE(SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END), 0) AS failed_jobs,
            (SELECT COUNT(*) FROM detections) AS detection_events,
            (SELECT COUNT(DISTINCT UPPER(TRIM(plate_text))) FROM detections WHERE plate_text IS NOT NULL AND TRIM(plate_text)!='') AS unique_plate_strings,
            (SELECT AVG(ocr_confidence) FROM detections WHERE ocr_confidence IS NOT NULL) AS average_ocr_confidence
            FROM jobs""").fetchone())


def get_analytics(database_path):
    with connect(database_path) as connection:
        daily = [dict(row) for row in connection.execute("""SELECT date(created_at) AS day, COUNT(*) AS jobs,
            SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS successful
            FROM jobs GROUP BY date(created_at) ORDER BY day DESC LIMIT 30""")]
        confidence = [dict(row) for row in connection.execute("""SELECT
            CASE WHEN ocr_confidence < .2 THEN '0-.19' WHEN ocr_confidence < .4 THEN '.2-.39'
            WHEN ocr_confidence < .6 THEN '.4-.59' WHEN ocr_confidence < .8 THEN '.6-.79' ELSE '.8-1' END AS bucket,
            COUNT(*) AS count FROM detections WHERE ocr_confidence IS NOT NULL GROUP BY bucket ORDER BY bucket""")]
        return {"daily": list(reversed(daily)), "confidence": confidence}
