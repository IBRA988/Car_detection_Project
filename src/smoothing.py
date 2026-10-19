"""
Speed smoothing filter applying Exponential Moving Average (EMA) and outlier suppression.
Prevents display jitter and ensures stable telemetry readings for lanes and vehicles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.schemas import TrackedVehicle


class SpeedSmoother:
    """Temporal smoothing engine for vehicle and lane speeds."""

    def __init__(self, lane_count: int, smooth_coef: float = 0.65) -> None:
        """
        Initialize the speed smoother.

        Args:
            lane_count: Number of lanes to monitor.
            smooth_coef: EMA weight for previous smoothed value (0.0 to 1.0).
                         Higher values produce smoother, slower-reacting curves.
        """
        self.lane_count = lane_count
        self.smooth_coef = max(0.0, min(1.0, float(smooth_coef)))

        self._lane_values = [0.0] * lane_count
        self._lane_initialized = [False] * lane_count

        # Vehicle-specific smoothing dictionary: track_id -> smoothed_speed
        self._vehicle_values: Dict[int, float] = {}

    def get_lane_speeds(self) -> List[float]:
        """Return the current list of smoothed lane speeds."""
        return list(self._lane_values)

    def smooth_lane_speeds(self, raw_lane_speeds: Sequence[float]) -> List[float]:
        """
        Apply EMA filter to lane speeds.

        If a lane has 0 km/h (e.g. no active vehicle), it slowly decays or retains
        its previous state according to EMA smoothing.
        """
        for i, raw_speed in enumerate(raw_lane_speeds):
            if i >= len(self._lane_values):
                break

            # If no car in lane, decay gently
            if raw_speed <= 0.0:
                if self._lane_initialized[i]:
                    self._lane_values[i] *= 0.95
                    if self._lane_values[i] < 2.0:
                        self._lane_values[i] = 0.0
                continue

            if not self._lane_initialized[i]:
                self._lane_values[i] = float(raw_speed)
                self._lane_initialized[i] = True
            else:
                self._lane_values[i] = (
                    self.smooth_coef * self._lane_values[i]
                    + (1.0 - self.smooth_coef) * float(raw_speed)
                )

        return self.get_lane_speeds()

    def smooth_vehicle(self, vehicle: TrackedVehicle) -> Optional[float]:
        """
        Smooth individual vehicle speed across consecutive frames.
        Updates vehicle.smoothed_speed_kmh in-place.
        """
        raw_speed = vehicle.current_speed_kmh
        if raw_speed is None:
            return None

        v_id = vehicle.track_id
        if v_id not in self._vehicle_values:
            self._vehicle_values[v_id] = raw_speed
        else:
            prev = self._vehicle_values[v_id]
            # EMA smoothing
            smoothed = self.smooth_coef * prev + (1.0 - self.smooth_coef) * raw_speed
            self._vehicle_values[v_id] = smoothed

        vehicle.smoothed_speed_kmh = self._vehicle_values[v_id]
        return vehicle.smoothed_speed_kmh

    def prune_stale_vehicles(self, active_track_ids: Sequence[int]) -> None:
        """Remove cached vehicle smoothing states for tracks that have ended."""
        active_set = set(active_track_ids)
        stale_ids = [v_id for v_id in self._vehicle_values if v_id not in active_set]
        for v_id in stale_ids:
            self._vehicle_values.pop(v_id, None)
