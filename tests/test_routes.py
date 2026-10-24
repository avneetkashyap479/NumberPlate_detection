import io


def test_health_and_empty_stats(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json["plate_model_available"] is False
    stats = client.get("/api/stats")
    assert stats.status_code == 200
    assert stats.json["detection_events"] == 0


def test_upload_validation_and_missing_model(client):
    missing = client.post("/api/detect/image", data={})
    assert missing.status_code == 400
    unsupported = client.post("/api/detect/image", data={"file": (io.BytesIO(b"nope"), "note.txt")})
    assert unsupported.status_code == 400
    from PIL import Image
    image = Image.new("RGB", (24, 24), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = client.post("/api/detect/image", data={"file": (io.BytesIO(buffer.getvalue()), "input.png")}, content_type="multipart/form-data")
    assert result.status_code == 503
    assert "weights were not found" in result.json["error"]
    history = client.get("/api/history")
    assert history.json["total"] == 1
    assert history.json["items"][0]["status"] == "failed"


def test_missing_history_and_pages(client):
    assert client.get("/api/history/1234").status_code == 404
    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/analytics").status_code == 200
    image_page = client.get("/image")
    video_page = client.get("/video")
    assert image_page.status_code == 200
    assert video_page.status_code == 200
    assert b"disabled" in image_page.data
    assert b"disabled" in video_page.data


def test_active_model_health_and_inference(tmp_path):
    from pathlib import Path
    from app import create_app

    root = Path(__file__).resolve().parent.parent
    best_pt = root / "models" / "best.pt"
    if not best_pt.is_file():
        return

    roots = {name: tmp_path / name.lower() for name in ("UPLOAD_IMAGE_DIR", "UPLOAD_VIDEO_DIR", "ANNOTATED_DIR", "CROPS_DIR", "VIDEO_OUTPUT_DIR")}
    for p in roots.values():
        p.mkdir()

    TestSettings = type("ActiveTestSettings", (), {
        **roots,
        "MODEL_PATH": best_pt,
        "CONFIDENCE_THRESHOLD": 0.25,
        "FRAME_INTERVAL": 5,
        "MAX_VIDEO_SECONDS": 60,
        "OCR_LANGUAGES": ["en"],
        "DATABASE_PATH": tmp_path / "active_test.sqlite3",
    })
    app = create_app({
        **roots,
        "DATABASE_PATH": TestSettings.DATABASE_PATH,
        "MODEL_PATH": TestSettings.MODEL_PATH,
        "SETTINGS": TestSettings,
        "TESTING": True,
        "MAX_CONTENT_LENGTH": 100 * 1024 * 1024
    })
    client = app.test_client()

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json["plate_model_available"] is True

    image_page = client.get("/image")
    assert image_page.status_code == 200
    assert b"Run plate detection" in image_page.data

    sample_img = root / "samples" / "car_sample_1.jpg"
    if sample_img.is_file():
        with open(sample_img, "rb") as f:
            res = client.post("/api/detect/image", data={"file": (f, "sample.jpg")}, content_type="multipart/form-data")
        assert res.status_code == 200
        assert res.json["ok"] is True
        assert len(res.json["detections"]) > 0

        job_id = res.json["job_id"]
        job_page = client.get(f"/job/{job_id}")
        assert job_page.status_code == 200

        stats = client.get("/api/stats")
        assert stats.status_code == 200
        assert stats.json["images_processed"] == 1
        assert stats.json["detection_events"] >= 1
