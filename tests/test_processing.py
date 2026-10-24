import pytest

from detection.evaluation import character_accuracy, detection_metrics
from detection.plate_detector import ModelUnavailableError, PlateDetector


def test_missing_model_fails_explicitly(tmp_path):
    detector = PlateDetector(tmp_path / "missing.pt")
    with pytest.raises(ModelUnavailableError, match="Plate detector weights were not found"):
        detector.detect(None)


def test_detection_metrics_match_each_box_once():
    result = detection_metrics([(0, 0, 10, 10), (1, 1, 9, 9)], [(0, 0, 10, 10)], .5)
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (1, 1, 0)
    assert result["precision"] == .5
    assert result["recall"] == 1.0


def test_character_accuracy():
    assert character_accuracy("AB12", "AB12") == 1
    assert character_accuracy("AB1", "AB12") == .75
