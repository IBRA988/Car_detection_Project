"""
Configuration manager for the traffic analysis pipeline.
Supports loading from YAML, dataclass validation, and command-line overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from src.schemas import CountingGate, LaneConfig


@dataclass
class ModelConfig:
    weights: str = "yolov8n.pt"
    confidence_threshold: float = 0.30
    iou_threshold: float = 0.45
    device: str = "auto"
    target_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    class_names: Dict[int, str] = field(
        default_factory=lambda: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    )


@dataclass
class VideoConfig:
    input_path: str = "4K_Video_of_Highway_Traffic_720p.mp4"
    output_path: Optional[str] = "output/analyzed_highway.mp4"
    mask_path: Optional[str] = "full mask.png"
    target_fps: float = 30.0
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


@dataclass
class TrackerConfig:
    max_age: int = 25
    min_hits: int = 3
    iou_threshold: float = 0.30
    track_history_length: int = 30


@dataclass
class EstimatorConfig:
    default_px_to_meter: float = 0.055
    min_displacement_px: float = 1.5
    max_speed_kmh: float = 220.0
    min_speed_kmh: float = 5.0
    speed_smoothing_factor: float = 0.65


@dataclass
class VisualizationConfig:
    show_lanes: bool = False
    show_lane_speeds: bool = False
    show_trails: bool = True
    show_bounding_boxes: bool = True
    show_speed_badges: bool = True
    show_counting_lines: bool = True
    single_line_gates: bool = True
    show_hud_header: bool = True
    lane_alpha: float = 0.20
    trail_thickness: int = 2
    box_thickness: int = 2


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    lanes: List[LaneConfig] = field(default_factory=list)
    counting_gates: Dict[str, CountingGate] = field(default_factory=dict)


def _compute_text_point(polygon: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Helper to locate top-center position for lane speed text."""
    if not polygon:
        return (0, 0)
    sorted_by_y = sorted(polygon, key=lambda p: p[1])
    top_two = sorted_by_y[:2]
    cx = int(sum(p[0] for p in top_two) / len(top_two)) - 25
    cy = int(min(p[1] for p in top_two)) - 8
    return (max(0, cx), max(10, cy))


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and validate configuration from YAML, applying defaults where omitted."""
    raw_cfg: Dict[str, Any] = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

    model_data = raw_cfg.get("model", {})
    video_data = raw_cfg.get("video", {})
    tracker_data = raw_cfg.get("tracker", {})
    estimator_data = raw_cfg.get("estimator", {})
    vis_data = raw_cfg.get("visualization", {})

    model = ModelConfig(
        weights=str(model_data.get("weights", "yolov8n.pt")),
        confidence_threshold=float(model_data.get("confidence_threshold", 0.30)),
        iou_threshold=float(model_data.get("iou_threshold", 0.45)),
        device=str(model_data.get("device", "auto")),
        target_classes=list(model_data.get("target_classes", [2, 3, 5, 7])),
        class_names={int(k): str(v) for k, v in model_data.get("class_names", {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}).items()}
    )

    video = VideoConfig(
        input_path=str(video_data.get("input_path", "4K_Video_of_Highway_Traffic_720p.mp4")),
        output_path=video_data.get("output_path", "output/analyzed_highway.mp4"),
        mask_path=video_data.get("mask_path", "full mask.png"),
        target_fps=float(video_data.get("target_fps", 30.0)),
        resize_width=video_data.get("resize_width"),
        resize_height=video_data.get("resize_height"),
    )

    tracker = TrackerConfig(
        max_age=int(tracker_data.get("max_age", 25)),
        min_hits=int(tracker_data.get("min_hits", 3)),
        iou_threshold=float(tracker_data.get("iou_threshold", 0.30)),
        track_history_length=int(tracker_data.get("track_history_length", 30)),
    )

    estimator = EstimatorConfig(
        default_px_to_meter=float(estimator_data.get("default_px_to_meter", 0.055)),
        min_displacement_px=float(estimator_data.get("min_displacement_px", 1.5)),
        max_speed_kmh=float(estimator_data.get("max_speed_kmh", 220.0)),
        min_speed_kmh=float(estimator_data.get("min_speed_kmh", 5.0)),
        speed_smoothing_factor=float(estimator_data.get("speed_smoothing_factor", 0.65)),
    )

    visualization = VisualizationConfig(
        show_lanes=bool(vis_data.get("show_lanes", False)),
        show_lane_speeds=bool(vis_data.get("show_lane_speeds", False)),
        show_trails=bool(vis_data.get("show_trails", True)),
        show_bounding_boxes=bool(vis_data.get("show_bounding_boxes", True)),
        show_speed_badges=bool(vis_data.get("show_speed_badges", True)),
        show_counting_lines=bool(vis_data.get("show_counting_lines", True)),
        single_line_gates=bool(vis_data.get("single_line_gates", True)),
        show_hud_header=bool(vis_data.get("show_hud_header", True)),
        lane_alpha=float(vis_data.get("lane_alpha", 0.20)),
        trail_thickness=int(vis_data.get("trail_thickness", 2)),
        box_thickness=int(vis_data.get("box_thickness", 2)),
    )

    lanes: List[LaneConfig] = []
    raw_lanes = raw_cfg.get("lanes", {})
    all_lane_dicts = []
    if isinstance(raw_lanes, dict):
        all_lane_dicts.extend(raw_lanes.get("left", []))
        all_lane_dicts.extend(raw_lanes.get("right", []))
    elif isinstance(raw_lanes, list):
        all_lane_dicts.extend(raw_lanes)

    for item in all_lane_dicts:
        poly = [tuple(p) for p in item["polygon"]]
        color = tuple(item.get("color", [255, 255, 255]))
        lane_cfg = LaneConfig(
            name=str(item.get("name", "Lane")),
            polygon=poly,
            px_to_meter=float(item.get("px_to_meter", estimator.default_px_to_meter)),
            color=color,
            direction=str(item.get("direction", "Unknown")),
            text_point=_compute_text_point(poly),
        )
        lanes.append(lane_cfg)

    gates: Dict[str, CountingGate] = {}
    raw_gates = raw_cfg.get("counting_gates", {})
    for key, g_item in raw_gates.items():
        gates[key] = CountingGate(
            name=str(g_item.get("name", key)),
            line_up=tuple(g_item["line_up"]),
            line_down=tuple(g_item["line_down"]),
            color=tuple(g_item.get("color", (0, 255, 0))),
        )

    return AppConfig(
        model=model,
        video=video,
        tracker=tracker,
        estimator=estimator,
        visualization=visualization,
        lanes=lanes,
        counting_gates=gates,
    )
