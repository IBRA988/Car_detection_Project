"""Unit tests for YOLOv8 VehicleDetector."""

import os
import numpy as np
import pytest
from src.detector import VehicleDetector


def test_detector_initialization():
    weights = "yolov8n.pt"
    if not os.path.exists(weights):
        pytest.skip("yolov8n.pt not present locally")

    detector = VehicleDetector(
        weights_path=weights,
        confidence_threshold=0.35,
        target_classes=[2, 3, 5, 7],
        device="cpu",
    )
    assert detector.confidence_threshold == 0.35
    assert 2 in detector.target_classes
    assert detector.device == "cpu"


def test_detector_empty_frame():
    weights = "yolov8n.pt"
    if not os.path.exists(weights):
        pytest.skip("yolov8n.pt not present locally")

    detector = VehicleDetector(weights_path=weights, device="cpu")
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    detections = detector.detect(empty_frame)
    assert detections == []


def test_detector_black_frame():
    weights = "yolov8n.pt"
    if not os.path.exists(weights):
        pytest.skip("yolov8n.pt not present locally")

    detector = VehicleDetector(weights_path=weights, device="cpu")
    # Black frame has no vehicles
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0
