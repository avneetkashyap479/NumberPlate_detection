import cv2
import numpy as np


class OCREngine:
    def __init__(self, languages=None):
        self.languages = languages or ["en"]
        self._reader = None

    def _load(self):
        try:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=False)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize EasyOCR: {exc}") from exc

    def read(self, crop):
        if self._reader is None:
            self._load()
        candidates = [crop]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        candidates.extend([enlarged, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(enlarged)])
        best = ("", 0.0)
        for candidate in candidates:
            try:
                rows = self._reader.readtext(candidate, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
                for _, text, confidence in rows:
                    if float(confidence) > best[1]:
                        best = (str(text), float(confidence))
            except Exception:
                continue
        raw = best[0]
        return {"raw_ocr_text": raw, "plate_text": raw.strip().upper() if raw else None, "ocr_confidence": best[1] if raw else None}
