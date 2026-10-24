from database import db


def test_job_detection_stats_and_cascade(tmp_path):
    path = tmp_path / "anpr.sqlite3"
    db.initialize_database(path)
    job_id = db.create_job(path, "image", "street.jpg", tmp_path / "street.jpg")
    detection_id = db.add_detection(path, job_id, {"plate_text": "AB12CD3456", "raw_ocr_text": "ab12cd3456",
        "detection_confidence": .91, "ocr_confidence": .8, "crop_path": tmp_path / "crop.jpg"})
    assert detection_id > 0
    db.finish_job(path, job_id, "completed", .42, tmp_path / "annotated.jpg")
    job = db.get_job(path, job_id)
    assert job["detections"][0]["raw_ocr_text"] == "ab12cd3456"
    stats = db.get_stats(path)
    assert stats["images_processed"] == 1
    assert stats["detection_events"] == 1
    assert stats["unique_plate_strings"] == 1
    assert db.delete_job(path, job_id)
    assert db.get_job(path, job_id) is None
