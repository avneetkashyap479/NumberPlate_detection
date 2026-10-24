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
