PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL CHECK(job_type IN ('image', 'video')),
    original_filename TEXT NOT NULL,
    input_path TEXT NOT NULL,
    output_path TEXT,
    status TEXT NOT NULL CHECK(status IN ('processing', 'completed', 'failed')),
    error_message TEXT,
    processing_seconds REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    plate_text TEXT,
    raw_ocr_text TEXT,
    detection_confidence REAL NOT NULL,
    ocr_confidence REAL,
    crop_path TEXT NOT NULL,
    frame_number INTEGER,
    timestamp_seconds REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_detections_job ON detections(job_id);
CREATE INDEX IF NOT EXISTS idx_detections_text ON detections(plate_text);
