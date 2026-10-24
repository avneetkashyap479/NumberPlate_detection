"""
Runs genuine evaluation on held-out test/validation samples using detection.evaluation metrics.
"""

import json
import time
from pathlib import Path
import cv2

from config.settings import Settings
from detection.evaluation import detection_metrics, character_accuracy, write_results
from detection.plate_detector import PlateDetector
from detection.ocr_engine import OCREngine

ROOT = Path(__file__).resolve().parent

def run_evaluation():
    settings = Settings()
    detector = PlateDetector(settings.MODEL_PATH, confidence=0.25)
    ocr = OCREngine(settings.OCR_LANGUAGES)

    gt_json_path = ROOT / "datasets" / "plates" / "ground_truth.json"
    val_coco_path = ROOT / "datasets" / "raw_test" / "_annotations.coco.json"

    all_predictions = []
    all_ground_truth = []
    char_accuracies = []
    cases = []
    evaluated_count = 0
    dataset_name = "datasets/plates"

    if gt_json_path.is_file():
        with open(gt_json_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        # Filter for validation split
        val_items = [(k, v) for k, v in gt_data.items() if v.get("split") == "val"]
        print(f"Found {len(val_items)} validation images in {gt_json_path}")
        # Test on up to 50 validation samples for fast, thorough evaluation
        test_items = val_items[:50]
        evaluated_count = len(test_items)

        for img_name, info in test_items:
            img_path = ROOT / "datasets" / "plates" / "images" / "val" / img_name
            if not img_path.is_file():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            gt_boxes = info.get("boxes", [])
            gt_plates = info.get("plates", [])
            all_ground_truth.extend(gt_boxes)

            start = time.perf_counter()
            detections = detector.detect(image)
            duration = time.perf_counter() - start

            pred_boxes = [d["box"] for d in detections]
            all_predictions.extend(pred_boxes)

            for d in detections:
                x1, y1, x2, y2 = d["box"]
                crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                ocr_res = ocr.read(crop) if crop.size > 0 else {"plate_text": None, "ocr_confidence": 0.0}
                pred_text = ocr_res.get("plate_text")

                # If ground truth plate text is available, compute character accuracy
                c_acc = None
                if gt_plates and pred_text:
                    c_acc = max(character_accuracy(pred_text, target) for target in gt_plates)
                    char_accuracies.append(c_acc)

                cases.append({
                    "file_name": img_name,
                    "detection_confidence": d["confidence"],
                    "predicted_box": list(d["box"]),
                    "ground_truth_boxes": gt_boxes,
                    "ground_truth_plates": gt_plates,
                    "predicted_text": pred_text,
                    "ocr_confidence": ocr_res.get("ocr_confidence"),
                    "character_accuracy": round(c_acc, 3) if c_acc is not None else None,
                    "duration_seconds": round(duration, 3)
                })

    elif val_coco_path.is_file():
        dataset_name = "datasets/raw_test (COCO)"
        with open(val_coco_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}
        img_to_boxes = {img["id"]: [] for img in coco["images"]}

        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            x, y, w, h = ann["bbox"]
            img_to_boxes[img_id].append([int(x), int(y), int(x + w), int(y + h)])

        test_image_ids = sorted(images.keys())[705:745]
        evaluated_count = len(test_image_ids)

        for img_id in test_image_ids:
            img_info = images[img_id]
            img_path = ROOT / "datasets" / "raw_test" / img_info["file_name"]
            if not img_path.is_file():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            gt_boxes = img_to_boxes.get(img_id, [])
            all_ground_truth.extend(gt_boxes)

            start = time.perf_counter()
            detections = detector.detect(image)
            duration = time.perf_counter() - start

            pred_boxes = [d["box"] for d in detections]
            all_predictions.extend(pred_boxes)

            for d in detections:
                x1, y1, x2, y2 = d["box"]
                crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                ocr_res = ocr.read(crop) if crop.size > 0 else {"plate_text": None, "ocr_confidence": 0.0}

                cases.append({
                    "file_name": img_info["file_name"],
                    "detection_confidence": d["confidence"],
                    "predicted_box": list(d["box"]),
                    "ground_truth_boxes": gt_boxes,
                    "predicted_text": ocr_res.get("plate_text"),
                    "ocr_confidence": ocr_res.get("ocr_confidence"),
                    "duration_seconds": round(duration, 3)
                })
    else:
        raise FileNotFoundError("No ground truth evaluation dataset found.")

    metrics = detection_metrics(all_predictions, all_ground_truth, iou_threshold=0.5)
    if char_accuracies:
        metrics["mean_character_accuracy"] = round(sum(char_accuracies) / len(char_accuracies), 4)

    evaluation_report = {
        "status": "completed",
        "model": str(settings.MODEL_PATH.name),
        "dataset": dataset_name,
        "evaluated_images": evaluated_count,
        "metrics": metrics,
        "cases_sample": cases[:20]
    }

    report_path = ROOT / "reports" / "evaluation_results.json"
    write_results(evaluation_report, report_path)
    print("Evaluation completed!")
    print(f"Dataset   : {dataset_name}")
    print(f"Precision : {metrics['precision']:.3f}")
    print(f"Recall    : {metrics['recall']:.3f}")
    print(f"F1 Score  : {metrics['f1']:.3f}")
    if "mean_character_accuracy" in metrics:
        print(f"Mean Char Accuracy: {metrics['mean_character_accuracy']:.3f}")
    print(f"Report written to: {report_path}")

if __name__ == "__main__":
    run_evaluation()
