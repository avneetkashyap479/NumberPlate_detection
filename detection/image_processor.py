import time
from pathlib import Path

import cv2

from detection.plate_detector import PlateDetector
from detection.ocr_engine import OCREngine


def process_image(image_path, job_id, settings, detector=None, ocr=None):
    started = time.perf_counter()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")
    detector = detector or PlateDetector(settings.MODEL_PATH, settings.CONFIDENCE_THRESHOLD)
    ocr = ocr or OCREngine(settings.OCR_LANGUAGES)
    found = detector.detect(image)
    height, width = image.shape[:2]
    results = []
    annotated = image.copy()
    for index, item in enumerate(found, start=1):
        x1, y1, x2, y2 = item["box"]
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        crop_name = f"{job_id}_{index}.jpg"
        crop_path = Path(settings.CROPS_DIR) / crop_name
        if not cv2.imwrite(str(crop_path), crop):
            raise RuntimeError("Could not save a plate crop.")
        ocr_result = ocr.read(crop)
        record = {**ocr_result, "detection_confidence": item["confidence"], "crop_path": str(crop_path), "box": [x1, y1, x2, y2]}
        results.append(record)
        label = f"{record['plate_text'] or 'Unreadable'} | det {item['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (41, 219, 183), 2)
        cv2.putText(annotated, label[:48], (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (41, 219, 183), 2, cv2.LINE_AA)
    output_path = Path(settings.ANNOTATED_DIR) / f"{job_id}_annotated.jpg"
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError("Could not save the annotated image.")
    return results, str(output_path), time.perf_counter() - started
