"""Unit tests for SORT tracker and Kalman filtering."""

import numpy as np
import pytest
from src.schemas import Detection
from src.tracker import KalmanBoxTracker, SortTracker, convert_bbox_to_z, convert_x_to_bbox


def test_bbox_conversions():
    bbox = (10, 20, 50, 60)  # w=40, h=40, cx=30, cy=40, s=1600, r=1.0
    z = convert_bbox_to_z(bbox)
    assert z[0, 0] == 30.0
    assert z[1, 0] == 40.0
    assert z[2, 0] == 1600.0
    assert z[3, 0] == 1.0

    recovered = convert_x_to_bbox(z.flatten())
    assert recovered == bbox


def test_kalman_tracker_lifecycle():
    KalmanBoxTracker.reset_counter()
    tracker = KalmanBoxTracker((10, 10, 30, 30), class_id=2, class_name="car")

    assert tracker.id == 1
    assert tracker.hits == 0

    # Predict
    predicted = tracker.predict()
    assert len(predicted) == 4

    # Update
    tracker.update((12, 12, 32, 32), confidence=0.95)
    assert tracker.hits == 1
    assert tracker.time_since_update == 0


def test_sort_tracker_association():
    KalmanBoxTracker.reset_counter()
    tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)

    # Frame 1: Single car detected
    det1 = [Detection(bbox=(100, 100, 150, 150), confidence=0.9, class_id=2, class_name="car")]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    assert tracks1[0].track_id == 1
    assert tracks1[0].class_name == "car"

    # Frame 2: Car shifted slightly
    det2 = [Detection(bbox=(105, 105, 155, 155), confidence=0.92, class_id=2, class_name="car")]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == 1  # ID preserved!
    assert len(tracks2[0].history) == 2
