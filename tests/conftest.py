import pytest

from app import create_app


@pytest.fixture
def app(tmp_path):
    roots = {name: tmp_path / name.lower() for name in ("UPLOAD_IMAGE_DIR", "UPLOAD_VIDEO_DIR", "ANNOTATED_DIR", "CROPS_DIR", "VIDEO_OUTPUT_DIR")}
    for path in roots.values():
        path.mkdir()
    TestSettings = type("TestSettings", (), {
        **roots,
        "MODEL_PATH": tmp_path / "missing_plate_model.pt",
        "CONFIDENCE_THRESHOLD": 0.25,
        "FRAME_INTERVAL": 5,
        "MAX_VIDEO_SECONDS": 60,
        "OCR_LANGUAGES": ["en"],
        "DATABASE_PATH": tmp_path / "test.sqlite3",
    })
    config = {**roots, "DATABASE_PATH": TestSettings.DATABASE_PATH, "MODEL_PATH": TestSettings.MODEL_PATH,
              "SETTINGS": TestSettings, "TESTING": True, "MAX_CONTENT_LENGTH": 1024 * 1024}
    return create_app(config)


@pytest.fixture
def client(app):
    return app.test_client()
