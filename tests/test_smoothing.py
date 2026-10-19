"""Unit tests for speed smoothing (EMA)."""

import pytest
from src.schemas import TrackedVehicle
from src.smoothing import SpeedSmoother


def test_lane_speed_smoothing():
    smoother = SpeedSmoother(lane_count=2, smooth_coef=0.5)

    # Initial update: initializes directly
    speeds1 = smoother.smooth_lane_speeds([100.0, 80.0])
    assert pytest.approx(speeds1[0], 0.1) == 100.0
    assert pytest.approx(speeds1[1], 0.1) == 80.0

    # Step 2: raw speed changes to 120.0
    # Expected: 0.5 * 100 + 0.5 * 120 = 110.0
    speeds2 = smoother.smooth_lane_speeds([120.0, 80.0])
    assert pytest.approx(speeds2[0], 0.1) == 110.0
    assert pytest.approx(speeds2[1], 0.1) == 80.0


def test_vehicle_speed_smoothing():
    smoother = SpeedSmoother(lane_count=1, smooth_coef=0.7)

    vehicle = TrackedVehicle(
        track_id=10,
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        class_id=2,
        class_name="car",
        current_speed_kmh=100.0,
    )

    # First update
    smoother.smooth_vehicle(vehicle)
    assert vehicle.smoothed_speed_kmh == 100.0

    # Second update with speed = 120.0
    # Expected: 0.7 * 100.0 + 0.3 * 120.0 = 70 + 36 = 106.0
    vehicle.current_speed_kmh = 120.0
    smoother.smooth_vehicle(vehicle)
    assert pytest.approx(vehicle.smoothed_speed_kmh, 0.1) == 106.0
