from pathlib import Path
import os
import secrets

from dotenv import load_dotenv
import certifi

if "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = certifi.where()

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings:
    SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
    _db_env = os.getenv("DATABASE_PATH", "database/anpr.sqlite3")
    DATABASE_PATH = Path(_db_env) if Path(_db_env).is_absolute() else (ROOT / _db_env).resolve()
    _model_env = os.getenv("PLATE_MODEL_PATH", "models/best.pt")
    MODEL_PATH = Path(_model_env) if Path(_model_env).is_absolute() else (ROOT / _model_env).resolve()
    UPLOAD_IMAGE_DIR = ROOT / "uploads" / "images"
    UPLOAD_VIDEO_DIR = ROOT / "uploads" / "videos"
    ANNOTATED_DIR = ROOT / "output" / "annotated"
    CROPS_DIR = ROOT / "output" / "crops"
    VIDEO_OUTPUT_DIR = ROOT / "output" / "videos"
    CONFIDENCE_THRESHOLD = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
    FRAME_INTERVAL = int(os.getenv("VIDEO_FRAME_INTERVAL", "15"))
    MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "180"))
    OCR_LANGUAGES = [value.strip() for value in os.getenv("OCR_LANGUAGES", "en").split(",") if value.strip()]


def ensure_directories(settings=Settings):
    for name in ("UPLOAD_IMAGE_DIR", "UPLOAD_VIDEO_DIR", "ANNOTATED_DIR", "CROPS_DIR", "VIDEO_OUTPUT_DIR"):
        Path(getattr(settings, name)).mkdir(parents=True, exist_ok=True)
    Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
