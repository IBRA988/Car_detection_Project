"""Unit tests for vehicle counting gates and bilateral flows."""

import pytest
from src.counter import VehicleCounter
from src.schemas import CountingGate, TrackedVehicle


@pytest.fixture
def test_counter():
    gate_left = CountingGate(
        name="Northbound Gate",
        line_up=(100, 200, 300, 200),
        line_down=(100, 300, 300, 300),
        color=(0, 255, 0),
    )
    return VehicleCounter({"left": gate_left})


def test_vehicle_counting_and_debounce(test_counter):
    # Vehicle inside counting zone: x=200, y=250
    v1 = TrackedVehicle(
        track_id=1,
        bbox=(180, 230, 220, 270),  # centroid = (200, 250)
        confidence=0.9,
        class_id=2,
        class_name="car",
    )

    counts = test_counter.update([v1])
    assert counts["left"] == 1
    assert v1.is_counted is True
    assert test_counter.get_total_count() == 1
    assert test_counter.class_breakdown["car"] == 1

    # Second frame: same vehicle still in zone -> should NOT recount
    counts2 = test_counter.update([v1])
    assert counts2["left"] == 1
    assert test_counter.get_total_count() == 1


def test_vehicle_outside_zone_not_counted(test_counter):
    # Outside zone: x=50, y=50
    v_out = TrackedVehicle(
        track_id=2,
        bbox=(40, 40, 60, 60),
        confidence=0.85,
        class_id=7,
        class_name="truck",
    )

    counts = test_counter.update([v_out])
    assert counts["left"] == 0
    assert v_out.is_counted is False
    assert test_counter.get_total_count() == 0
