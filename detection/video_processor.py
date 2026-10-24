import time
from pathlib import Path

import cv2

from detection.plate_detector import PlateDetector
from detection.ocr_engine import OCREngine


def process_video(video_path, job_id, settings, detector=None, ocr=None):
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0
    if duration > settings.MAX_VIDEO_SECONDS:
        capture.release()
        raise ValueError(f"Video exceeds the {settings.MAX_VIDEO_SECONDS}-second limit.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(settings.VIDEO_OUTPUT_DIR) / f"{job_id}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    detector = detector or PlateDetector(settings.MODEL_PATH, settings.CONFIDENCE_THRESHOLD)
    ocr = ocr or OCREngine(settings.OCR_LANGUAGES)
    detections, frame_number = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number % max(1, settings.FRAME_INTERVAL) == 0:
                found = detector.detect(frame)
                for index, item in enumerate(found, start=1):
                    x1, y1, x2, y2 = item["box"]
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    crop_path = Path(settings.CROPS_DIR) / f"{job_id}_f{frame_number}_{index}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    detections.append({**ocr.read(crop), "detection_confidence": item["confidence"],
                        "crop_path": str(crop_path), "frame_number": frame_number,
                        "timestamp_seconds": frame_number / fps})
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (41, 219, 183), 2)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return detections, str(output_path), time.perf_counter() - started
