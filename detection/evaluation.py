import json
from pathlib import Path


def intersection_over_union(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def detection_metrics(predictions, ground_truth, iou_threshold=0.5):
    candidates = sorted(((intersection_over_union(p, g), pi, gi)
                        for pi, p in enumerate(predictions) for gi, g in enumerate(ground_truth)), reverse=True)
    used_predictions, used_truth = set(), set()
    for overlap, prediction_index, truth_index in candidates:
        if overlap < iou_threshold:
            break
        if prediction_index not in used_predictions and truth_index not in used_truth:
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
    tp = len(used_predictions)
    fp, fn = len(predictions) - tp, len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"iou_threshold": iou_threshold, "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def character_accuracy(predicted, expected):
    left, right = predicted.upper(), expected.upper()
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (char_left != char_right)))
        previous = current
    return 1.0 - previous[-1] / max(1, len(left), len(right))


def write_results(results, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path
