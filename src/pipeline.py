"""
Main orchestration pipeline for the traffic surveillance and vehicle analysis system.
Binds detection, tracking, speed estimation, counting, and visual telemetry together.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np

from src.config import AppConfig
from src.counter import VehicleCounter
from src.detector import VehicleDetector
from src.estimator import SpeedEstimator
from src.schemas import PipelineMetrics, TrackedVehicle
from src.smoothing import SpeedSmoother
from src.tracker import SortTracker
from src.utils import load_and_prep_mask
from src.visualizer import HUDVisualizer

logger = logging.getLogger(__name__)


class TrafficAnalysisPipeline:
    """Orchestrates end-to-end video processing, telemetry calculation, and rendering."""

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the traffic pipeline with an AppConfig configuration instance.
        """
        self.config = config

        # 1. Initialize Sub-modules
        self.detector = VehicleDetector(
            weights_path=config.model.weights,
            confidence_threshold=config.model.confidence_threshold,
            iou_threshold=config.model.iou_threshold,
            target_classes=config.model.target_classes,
            class_name_map=config.model.class_names,
            device=config.model.device,
        )

        self.tracker = SortTracker(
            max_age=config.tracker.max_age,
            min_hits=config.tracker.min_hits,
            iou_threshold=config.tracker.iou_threshold,
            track_history_length=config.tracker.track_history_length,
        )

        self.estimator = SpeedEstimator(
            lanes=config.lanes,
            fps=config.video.target_fps,
            default_px_to_meter=config.estimator.default_px_to_meter,
            min_displacement_px=config.estimator.min_displacement_px,
            max_speed_kmh=config.estimator.max_speed_kmh,
            min_speed_kmh=config.estimator.min_speed_kmh,
        )

        self.smoother = SpeedSmoother(
            lane_count=len(config.lanes),
            smooth_coef=config.estimator.speed_smoothing_factor,
        )

        self.counter = VehicleCounter(gates=config.counting_gates)

        self.visualizer = HUDVisualizer(
            lanes=config.lanes,
            counting_gates=config.counting_gates,
            show_lanes=config.visualization.show_lanes,
            show_lane_speeds=config.visualization.show_lane_speeds,
            show_trails=config.visualization.show_trails,
            show_bounding_boxes=config.visualization.show_bounding_boxes,
            show_speed_badges=config.visualization.show_speed_badges,
            show_counting_lines=config.visualization.show_counting_lines,
            single_line_gates=config.visualization.single_line_gates,
            show_hud_header=config.visualization.show_hud_header,
            lane_alpha=config.visualization.lane_alpha,
            trail_thickness=config.visualization.trail_thickness,
            box_thickness=config.visualization.box_thickness,
        )

        self.mask: Optional[np.ndarray] = None
        self.metrics = PipelineMetrics()
        self._fps_history: List[float] = []

    def prepare_mask(self, width: int, height: int) -> None:
        """Load and scale the Region of Interest mask to the video resolution."""
        self.mask = load_and_prep_mask(self.config.video.mask_path, width, height)
        self.detector.set_roi_mask(self.mask)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, PipelineMetrics, List[TrackedVehicle]]:
        """
        Process a single video frame through the full detection, tracking, estimation, and visualization chain.

        Returns:
            Tuple of (annotated_frame, current_metrics, active_vehicles).
        """
        start_time = time.perf_counter()

        # 1. Detection
        det_start = time.perf_counter()
        detections = self.detector.detect(frame)
        det_time_ms = (time.perf_counter() - det_start) * 1000.0

        # 2. Tracking
        active_vehicles = self.tracker.update(detections)

        # 3. Speed Estimation & Smoothing
        raw_lane_speeds = self.estimator.estimate_all(active_vehicles)
        smoothed_lane_speeds = self.smoother.smooth_lane_speeds(raw_lane_speeds)

        for v in active_vehicles:
            self.smoother.smooth_vehicle(v)

        self.smoother.prune_stale_vehicles([v.track_id for v in active_vehicles])

        # 4. Counting & Gates
        gate_counts = self.counter.update(active_vehicles)
        active_gate_triggers = list(self.counter.active_gate_triggers)

        # 5. Telemetry & FPS Calculation
        elapsed = time.perf_counter() - start_time
        instant_fps = 1.0 / max(1e-5, elapsed)
        self._fps_history.append(instant_fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        self.metrics.frame_index += 1
        self.metrics.fps = avg_fps
        self.metrics.inference_time_ms = det_time_ms
        self.metrics.total_processing_time_ms = elapsed * 1000.0
        self.metrics.active_tracks = len(active_vehicles)
        self.metrics.count_left = gate_counts.get("left_gate", 0)
        self.metrics.count_right = gate_counts.get("right_gate", 0)
        self.metrics.total_counted = self.counter.get_total_count()
        self.metrics.lane_speeds = smoothed_lane_speeds

        # 6. Visualization & Rendering
        annotated_frame = self.visualizer.render(
            frame=frame,
            vehicles=active_vehicles,
            metrics=self.metrics,
            lane_speeds=smoothed_lane_speeds,
            active_gate_triggers=active_gate_triggers,
        )

        return annotated_frame, self.metrics, active_vehicles

    def run(
        self,
        show: bool = False,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[PipelineMetrics], None]] = None,
    ) -> PipelineMetrics:
        """
        Execute the analysis on the configured input video.

        Args:
            show: If True, display interactive window using cv2.imshow.
            max_frames: Optional frame limit for quick testing/benchmarking.
            progress_callback: Optional callback receiving updated PipelineMetrics per frame.

        Returns:
            Final PipelineMetrics summary.
        """
        input_path = self.config.video.input_path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file does not exist: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video source: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.estimator.fps = source_fps
        self.prepare_mask(width, height)

        writer: Optional[cv2.VideoWriter] = None
        output_path = self.config.video.output_path
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, source_fps, (width, height))
            logger.info(f"Writing annotated video to {output_path} ({width}x{height} @ {source_fps:.1f} FPS)")

        frame_count = 0
        window_name = "Traffic Vision Pro - Highway Surveillance"

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1
                if max_frames and frame_count > max_frames:
                    break

                annotated_frame, metrics, _ = self.process_frame(frame)

                if writer:
                    writer.write(annotated_frame)

                if progress_callback:
                    progress_callback(metrics)

                if show:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' or Esc to exit
                        logger.info("User requested exit.")
                        break

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

        logger.info(f"Pipeline finished. Processed {frame_count} frames. Total counted: {self.metrics.total_counted}")
        return self.metrics
