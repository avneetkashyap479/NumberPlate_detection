from pathlib import Path


class ModelUnavailableError(RuntimeError):
    pass


class PlateDetector:
    def __init__(self, model_path, confidence=0.25):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._model = None

    def _load(self):
        if not self.model_path.is_file():
            raise ModelUnavailableError(
                f"Plate detector weights were not found at {self.model_path}. Add a plate-specific YOLO .pt model and set PLATE_MODEL_PATH."
            )
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load configured plate model: {exc}") from exc

    def detect(self, image):
        if self._model is None:
            self._load()
        try:
            results = self._model.predict(source=image, conf=self.confidence, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"Plate model inference failed: {exc}") from exc
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append({"box": (x1, y1, x2, y2), "confidence": float(box.conf[0].item())})
        return detections
