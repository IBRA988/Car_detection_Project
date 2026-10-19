"""
Traffic Vision: Real-time Vehicle Detection, Tracking, and Speed Estimation.
Production Computer Vision pipeline using YOLOv8, SORT, and Geometric Speed Estimation.
"""

__version__ = "1.0.0"
__author__ = "Advanced Computer Vision Engineering"

from src.schemas import (
    BBox,
    CountingGate,
    Detection,
    LaneConfig,
    PipelineMetrics,
    Point,
    SpeedEstimate,
    TrackedVehicle,
)
from src.config import AppConfig, load_config
from src.detector import VehicleDetector
from src.tracker import SortTracker
from src.estimator import SpeedEstimator
from src.smoothing import SpeedSmoother
from src.counter import VehicleCounter
from src.visualizer import HUDVisualizer
from src.pipeline import TrafficAnalysisPipeline

__all__ = [
    "AppConfig",
    "load_config",
    "BBox",
    "Point",
    "Detection",
    "TrackedVehicle",
    "LaneConfig",
    "CountingGate",
    "SpeedEstimate",
    "PipelineMetrics",
    "VehicleDetector",
    "SortTracker",
    "SpeedEstimator",
    "SpeedSmoother",
    "VehicleCounter",
    "HUDVisualizer",
    "TrafficAnalysisPipeline",
]
