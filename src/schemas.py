"""
Data models and typed schemas for the vehicle detection and speed estimation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple, List, Optional

Point = Tuple[float, float]
IntPoint = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Detection:
    """Represents an object detected by the YOLO model in a single frame."""
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> IntPoint:
        """Calculate the integer center coordinate of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class TrackedVehicle:
    """Represents a vehicle tracked across consecutive video frames."""
    track_id: int
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    history: List[IntPoint] = field(default_factory=list)
    current_speed_kmh: Optional[float] = None
    smoothed_speed_kmh: Optional[float] = None
    lane_id: Optional[int] = None
    lane_name: Optional[str] = None
    direction: Optional[str] = None
    is_counted: bool = False

    @property
    def centroid(self) -> IntPoint:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2


@dataclass(frozen=True)
class LaneConfig:
    """Configuration for a monitored highway lane."""
    name: str
    polygon: List[IntPoint]
    px_to_meter: float
    color: Tuple[int, int, int]
    direction: str = "Unknown"
    text_point: Optional[IntPoint] = None


@dataclass
class CountingGate:
    """Bidirectional counting gate consisting of boundary lines."""
    name: str
    line_up: Tuple[int, int, int, int]  # x1, y1, x2, y2
    line_down: Tuple[int, int, int, int]
    color: Tuple[int, int, int]
    count: int = 0
    counted_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class SpeedEstimate:
    """Computed speed estimation details for an object."""
    vehicle_id: int
    speed_kmh: float
    displacement_px: float
    dt_seconds: float
    lane_index: Optional[int] = None


@dataclass
class PipelineMetrics:
    """Performance and telemetry metrics for the active video pipeline."""
    frame_index: int = 0
    fps: float = 0.0
    inference_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    active_tracks: int = 0
    count_left: int = 0
    count_right: int = 0
    total_counted: int = 0
    lane_speeds: List[float] = field(default_factory=list)
