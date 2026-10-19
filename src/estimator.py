"""
Speed estimation engine for highway traffic analysis.
Converts pixel displacement to metric speed (km/h) based on calibrated lane geometry and FPS.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.schemas import LaneConfig, SpeedEstimate, TrackedVehicle
from src.utils import euclidean_distance, point_in_polygon

logger = logging.getLogger(__name__)


class SpeedEstimator:
    """Estimates instantaneous and windowed metric vehicle speed using pixel-to-meter calibration."""

    def __init__(
        self,
        lanes: Sequence[LaneConfig],
        fps: float = 30.0,
        default_px_to_meter: float = 0.055,
        min_displacement_px: float = 1.5,
        max_speed_kmh: float = 220.0,
        min_speed_kmh: float = 5.0,
        displacement_window: int = 3,
    ) -> None:
        """
        Initialize the speed estimator.

        Args:
            lanes: List of monitored highway lane configurations.
            fps: Frame rate of the video source (frames per second).
            default_px_to_meter: Default pixel-to-meter conversion factor if not specified by lane.
            min_displacement_px: Threshold under which movement is treated as stationary jitter.
            max_speed_kmh: Upper physical speed clamp to filter tracking glitches.
            min_speed_kmh: Lower threshold below which speed is rounded to 0 km/h.
            displacement_window: Number of historical points to compute displacement across.
        """
        self.lanes = list(lanes)
        self.fps = max(1.0, float(fps))
        self.default_px_to_meter = float(default_px_to_meter)
        self.min_displacement_px = float(min_displacement_px)
        self.max_speed_kmh = float(max_speed_kmh)
        self.min_speed_kmh = float(min_speed_kmh)
        self.displacement_window = max(1, int(displacement_window))

        # Lane polygon arrays for fast point-in-polygon queries
        self._lane_polygons = [np.asarray(lane.polygon, dtype=np.int32) for lane in self.lanes]

    def find_lane(self, point: Tuple[int, int]) -> Optional[int]:
        """Identify which lane polygon encloses the given point."""
        for idx, poly in enumerate(self._lane_polygons):
            if point_in_polygon(point, poly):
                return idx
        return None

    def estimate_vehicle_speed(self, vehicle: TrackedVehicle) -> Optional[SpeedEstimate]:
        """
        Estimate speed in km/h for a single tracked vehicle based on its centroid history.

        Formula:
            v_kmh = (displacement_px / num_intervals) * px_to_meter * FPS * 3.6
        """
        history = vehicle.history
        if len(history) < 2:
            return None

        # Use recent window to mitigate inter-frame centroid noise
        window_size = min(len(history), self.displacement_window + 1)
        p_start = history[-window_size]
        p_end = history[-1]
        intervals = window_size - 1

        total_disp = euclidean_distance(p_start, p_end)
        avg_disp_per_frame = total_disp / intervals

        # Sub-pixel movement noise suppression
        if avg_disp_per_frame < self.min_displacement_px:
            vehicle.current_speed_kmh = 0.0
            return SpeedEstimate(
                vehicle_id=vehicle.track_id,
                speed_kmh=0.0,
                displacement_px=total_disp,
                dt_seconds=intervals / self.fps,
                lane_index=vehicle.lane_id,
            )

        # Determine lane and calibration coefficient
        lane_idx = self.find_lane(p_end)
        px_to_meter = (
            self.lanes[lane_idx].px_to_meter
            if (lane_idx is not None and lane_idx < len(self.lanes))
            else self.default_px_to_meter
        )

        # Physics: speed = (m / frame) * (frames / sec) * 3.6 (m/s -> km/h)
        speed_mps = avg_disp_per_frame * px_to_meter * self.fps
        speed_kmh = speed_mps * 3.6

        # Outlier rejection / clamp
        if speed_kmh > self.max_speed_kmh:
            speed_kmh = self.max_speed_kmh
        elif speed_kmh < self.min_speed_kmh:
            speed_kmh = 0.0

        vehicle.current_speed_kmh = speed_kmh
        if lane_idx is not None:
            vehicle.lane_id = lane_idx
            vehicle.lane_name = self.lanes[lane_idx].name
            vehicle.direction = self.lanes[lane_idx].direction

        return SpeedEstimate(
            vehicle_id=vehicle.track_id,
            speed_kmh=speed_kmh,
            displacement_px=total_disp,
            dt_seconds=intervals / self.fps,
            lane_index=lane_idx,
        )

    def estimate_all(self, vehicles: Sequence[TrackedVehicle]) -> List[float]:
        """
        Estimate speeds for all active tracked vehicles and compute lane average speeds.

        Returns:
            List of average speeds for each lane [lane_0_speed, lane_1_speed, ...].
        """
        lane_count = len(self.lanes)
        lane_speeds: List[List[float]] = [[] for _ in range(lane_count)]

        for v in vehicles:
            estimate = self.estimate_vehicle_speed(v)
            if estimate and estimate.lane_index is not None and estimate.speed_kmh > 0:
                lane_speeds[estimate.lane_index].append(estimate.speed_kmh)

        # Compute mean speed per lane (or 0 if empty)
        avg_lane_speeds: List[float] = []
        for speeds in lane_speeds:
            avg_lane_speeds.append(float(np.mean(speeds)) if speeds else 0.0)

        return avg_lane_speeds
