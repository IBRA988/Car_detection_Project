"""Unit tests for utility helpers."""

import numpy as np
import pytest
from src.utils import compute_iou, euclidean_distance, point_in_polygon, select_optimal_device


def test_euclidean_distance():
    p1 = (0, 0)
    p2 = (3, 4)
    assert euclidean_distance(p1, p2) == 5.0

    p3 = (10, 20)
    assert euclidean_distance(p3, p3) == 0.0


def test_compute_iou():
    box1 = (0, 0, 10, 10)  # area = 100
    box2 = (0, 0, 10, 10)  # exact match
    assert compute_iou(box1, box2) == 1.0

    box3 = (10, 10, 20, 20)  # touching but no overlap
    assert compute_iou(box1, box3) == 0.0

    box4 = (0, 0, 10, 5)   # half overlap (inter=50, union=100)
    assert pytest.approx(compute_iou(box1, box4), 0.01) == 0.5


def test_point_in_polygon():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]

    assert point_in_polygon((50, 50), square) is True
    assert point_in_polygon((0, 0), square) is True  # on boundary
    assert point_in_polygon((150, 50), square) is False


def test_select_optimal_device():
    dev = select_optimal_device("cpu")
    assert dev == "cpu"

    auto_dev = select_optimal_device("auto")
    assert auto_dev in ("cuda", "mps", "cpu")
