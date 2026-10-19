"""Unit tests for pipeline data schemas and models."""

import pytest
from src.schemas import BBox, Detection, LaneConfig, PipelineMetrics, TrackedVehicle


def test_detection_properties():
    bbox: BBox = (100, 200, 300, 400)
    det = Detection(bbox=bbox, confidence=0.88, class_id=2, class_name="car")

    assert det.centroid == (200, 300)
    assert det.width == 200
    assert det.height == 200
    assert det.area == 40000
    assert det.class_name == "car"
    assert det.confidence == 0.88


def test_tracked_vehicle_properties():
    vehicle = TrackedVehicle(
        track_id=42,
        bbox=(50, 60, 150, 160),
        confidence=0.92,
        class_id=7,
        class_name="truck",
        history=[(100, 100), (100, 110)],
    )

    assert vehicle.track_id == 42
    assert vehicle.centroid == (100, 110)
    assert vehicle.class_name == "truck"
    assert len(vehicle.history) == 2
    assert vehicle.is_counted is False


def test_lane_config():
    lane = LaneConfig(
        name="Test Lane",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        px_to_meter=0.05,
        color=(255, 0, 0),
        direction="Southbound",
    )
    assert lane.name == "Test Lane"
    assert lane.px_to_meter == 0.05
    assert len(lane.polygon) == 4
    assert lane.direction == "Southbound"


def test_pipeline_metrics():
    metrics = PipelineMetrics(
        frame_index=150,
        fps=29.8,
        active_tracks=8,
        count_left=12,
        count_right=19,
        total_counted=31,
    )
    assert metrics.frame_index == 150
    assert metrics.fps == 29.8
    assert metrics.total_counted == 31
