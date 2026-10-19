"""Unit tests for mathematical speed estimation and calibration."""

import pytest
from src.estimator import SpeedEstimator
from src.schemas import LaneConfig, TrackedVehicle


@pytest.fixture
def sample_lane():
    return LaneConfig(
        name="Express Lane",
        polygon=[(0, 0), (200, 0), (200, 400), (0, 400)],
        px_to_meter=0.05,
        color=(255, 0, 0),
        direction="Southbound",
    )


def test_speed_estimation_formula(sample_lane):
    # 30 FPS, px_to_meter = 0.05 m/px
    # Displacement = 20 pixels in 1 interval
    # Real distance = 20 * 0.05 = 1.0 meter
    # Speed = 1.0 m / (1/30 s) = 30.0 m/s = 30.0 * 3.6 = 108.0 km/h
    estimator = SpeedEstimator(lanes=[sample_lane], fps=30.0, displacement_window=1)

    vehicle = TrackedVehicle(
        track_id=1,
        bbox=(50, 100, 90, 140),
        confidence=0.9,
        class_id=2,
        class_name="car",
        history=[(70, 100), (70, 120)],  # dy = 20 px
    )

    estimate = estimator.estimate_vehicle_speed(vehicle)
    assert estimate is not None
    assert pytest.approx(estimate.speed_kmh, 0.1) == 108.0
    assert vehicle.current_speed_kmh is not None
    assert pytest.approx(vehicle.current_speed_kmh, 0.1) == 108.0


def test_speed_noise_filtering(sample_lane):
    estimator = SpeedEstimator(
        lanes=[sample_lane],
        fps=30.0,
        min_displacement_px=2.0,
        min_speed_kmh=5.0,
    )

    # Stationary jitter: displacement = 0.5 px
    vehicle = TrackedVehicle(
        track_id=2,
        bbox=(50, 100, 90, 140),
        confidence=0.9,
        class_id=2,
        class_name="car",
        history=[(70, 100), (70, 100)],
    )

    estimate = estimator.estimate_vehicle_speed(vehicle)
    assert estimate is not None
    assert estimate.speed_kmh == 0.0


def test_speed_max_clamp(sample_lane):
    estimator = SpeedEstimator(
        lanes=[sample_lane],
        fps=30.0,
        max_speed_kmh=200.0,
        displacement_window=1,
    )

    # Huge glitch jump: dy = 1000 px -> speed >> 200 km/h
    vehicle = TrackedVehicle(
        track_id=3,
        bbox=(50, 100, 90, 140),
        confidence=0.9,
        class_id=2,
        class_name="car",
        history=[(70, 100), (70, 1100)],
    )

    estimate = estimator.estimate_vehicle_speed(vehicle)
    assert estimate is not None
    assert estimate.speed_kmh == 200.0  # clamped to max
