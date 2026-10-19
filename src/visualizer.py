"""
Heads-Up Display (HUD) visualizer for traffic surveillance and telemetry.
Renders real-time analytics, clean counting lines, motion trajectories, and metrics dashboards.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.schemas import CountingGate, LaneConfig, PipelineMetrics, TrackedVehicle
from src.utils import draw_styled_label


class HUDVisualizer:
    """Renders professional overlays, glassmorphic HUD banners, and telemetry badges."""

    def __init__(
        self,
        lanes: Sequence[LaneConfig],
        counting_gates: Optional[Dict[str, CountingGate]] = None,
        show_lanes: bool = False,
        show_lane_speeds: bool = False,
        show_trails: bool = True,
        show_bounding_boxes: bool = True,
        show_speed_badges: bool = True,
        show_counting_lines: bool = True,
        single_line_gates: bool = True,
        show_hud_header: bool = True,
        lane_alpha: float = 0.20,
        trail_thickness: int = 2,
        box_thickness: int = 2,
    ) -> None:
        """
        Initialize HUD Visualizer.

        Args:
            lanes: Highway lane polygons and color themes.
            counting_gates: Directional counting gates.
            show_lanes: Whether to render shaded lane polygons on the road.
            show_lane_speeds: Whether to render static speed boxes above lanes.
            show_trails: Whether to render vehicle trajectory history.
            show_bounding_boxes: Whether to render vehicle bounding boxes.
            show_speed_badges: Whether to render speed badges attached to vehicles.
            show_counting_lines: Whether to render counting line(s).
            single_line_gates: Whether to render a single clean counting line per flow.
            show_hud_header: Whether to render the top glassmorphism HUD dashboard.
            lane_alpha: Transparency factor for filled lane overlays (0.0 to 1.0).
            trail_thickness: Thickness of vehicle motion trajectories.
            box_thickness: Thickness of vehicle bounding boxes.
        """
        self.lanes = list(lanes)
        self.counting_gates = counting_gates or {}
        self.show_lanes = bool(show_lanes)
        self.show_lane_speeds = bool(show_lane_speeds)
        self.show_trails = bool(show_trails)
        self.show_bounding_boxes = bool(show_bounding_boxes)
        self.show_speed_badges = bool(show_speed_badges)
        self.show_counting_lines = bool(show_counting_lines)
        self.single_line_gates = bool(single_line_gates)
        self.show_hud_header = bool(show_hud_header)
        self.lane_alpha = float(lane_alpha)
        self.trail_thickness = int(trail_thickness)
        self.box_thickness = int(box_thickness)

        # Precompute polygon arrays for quick rendering if enabled
        self._lane_polys = [np.asarray(l.polygon, dtype=np.int32) for l in self.lanes]

    def draw_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Draw semi-transparent lane polygons and boundaries."""
        if not self.lanes or not self.show_lanes:
            return frame

        overlay = frame.copy()
        for idx, lane in enumerate(self.lanes):
            poly = self._lane_polys[idx]
            cv2.fillPoly(overlay, [poly], color=lane.color)

        cv2.addWeighted(overlay, self.lane_alpha, frame, 1.0 - self.lane_alpha, 0, frame)

        for poly in self._lane_polys:
            cv2.polylines(frame, [poly], isClosed=True, color=(240, 240, 240), thickness=1, lineType=cv2.LINE_AA)

        return frame

    def draw_lane_speeds(self, frame: np.ndarray, lane_speeds: Sequence[float]) -> None:
        """Draw speed badges at the entrance or top-center of each lane."""
        if not self.show_lane_speeds:
            return

        for idx, lane in enumerate(self.lanes):
            speed = lane_speeds[idx] if idx < len(lane_speeds) else 0.0
            pt = lane.text_point or (lane.polygon[0][0], lane.polygon[0][1])
            text = f"{int(round(speed))} km/h"
            draw_styled_label(
                frame,
                text,
                pt,
                bg_color=(30, 30, 30),
                text_color=(0, 255, 255) if speed > 100 else (255, 255, 255),
                font_scale=0.45,
                thickness=1,
            )

    def draw_counting_gates(self, frame: np.ndarray, active_triggers: Sequence[str] = ()) -> None:
        """Draw clean counting line(s) with dynamic trigger lighting when crossed."""
        if not self.show_counting_lines:
            return

        for gate_key, gate in self.counting_gates.items():
            is_active = gate_key in active_triggers
            line_color = (0, 255, 127) if is_active else (0, 0, 255)  # Vibrant emerald green when triggered, else red
            thickness = 4 if is_active else 3

            if self.single_line_gates:
                # Render a single clean, prominent counting line across the highway
                p1 = (gate.line_down[0], gate.line_down[1])
                p2 = (gate.line_down[2], gate.line_down[3])
                cv2.line(frame, p1, p2, line_color, thickness, lineType=cv2.LINE_AA)

                # Small clean badge indicator near line start
                label_x = min(p1[0], p2[0]) + 10
                label_y = min(p1[1], p2[1]) - 8
                draw_styled_label(
                    frame,
                    f"COUNT LINE: {gate.name.upper()}",
                    (label_x, max(15, label_y)),
                    bg_color=(15, 15, 15),
                    text_color=line_color,
                    font_scale=0.38,
                    thickness=1,
                )
            else:
                # Dual boundary lines mode
                p1_up = (gate.line_up[0], gate.line_up[1])
                p2_up = (gate.line_up[2], gate.line_up[3])
                cv2.line(frame, p1_up, p2_up, line_color, thickness, lineType=cv2.LINE_AA)

                p1_down = (gate.line_down[0], gate.line_down[1])
                p2_down = (gate.line_down[2], gate.line_down[3])
                cv2.line(frame, p1_down, p2_down, line_color, thickness, lineType=cv2.LINE_AA)

    def draw_tracked_vehicles(self, frame: np.ndarray, vehicles: Sequence[TrackedVehicle]) -> None:
        """Render vehicle bounding boxes, centroids, motion trails, and telemetry badges."""
        if not self.show_bounding_boxes:
            return

        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            cx, cy = v.centroid

            # Dynamic box color: Green if counted, Cyan by default
            box_color = (46, 204, 113) if v.is_counted else (241, 196, 15)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, self.box_thickness, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), cv2.FILLED)

            # Draw motion trajectory trail
            if self.show_trails and len(v.history) >= 2:
                pts = np.asarray(v.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame,
                    [pts],
                    isClosed=False,
                    color=(255, 120, 0),
                    thickness=self.trail_thickness,
                    lineType=cv2.LINE_AA,
                )

            # Primary label: [ID] Class Conf%
            label = f"#{v.track_id} {v.class_name.upper()} {int(v.confidence * 100)}%"
            draw_styled_label(
                frame,
                label,
                (x1, max(20, y1 - 4)),
                bg_color=(20, 20, 20),
                text_color=(255, 255, 255),
                font_scale=0.42,
            )

            # Speed badge attached to vehicle
            if self.show_speed_badges:
                speed = v.smoothed_speed_kmh or v.current_speed_kmh
                if speed is not None and speed > 0:
                    speed_str = f"{int(round(speed))} km/h"
                    speed_color = (0, 69, 255) if speed > 120 else (0, 255, 127)  # Red if speeding, else green
                    draw_styled_label(
                        frame,
                        speed_str,
                        (x1, y2 + 18),
                        bg_color=(15, 15, 15),
                        text_color=speed_color,
                        font_scale=0.45,
                        thickness=1,
                    )

    def draw_hud_header(self, frame: np.ndarray, metrics: PipelineMetrics) -> None:
        """Render a glassmorphic top dashboard banner with key traffic intelligence metrics."""
        if not self.show_hud_header:
            return

        h, w = frame.shape[:2]
        header_h = 48

        # Create semi-transparent dark banner overlay
        sub_img = frame[0:header_h, 0:w]
        dark_rect = np.full_like(sub_img, 20)
        cv2.addWeighted(dark_rect, 0.75, sub_img, 0.25, 0, sub_img)
        frame[0:header_h, 0:w] = sub_img

        # Bottom accent line
        cv2.line(frame, (0, header_h), (w, header_h), (0, 200, 255), 2)

        # Telemetry metrics strings
        title_text = "TRAFFIC VISION PRO"
        left_text = f"LEFT FLOW: {metrics.count_left}"
        right_text = f"RIGHT FLOW: {metrics.count_right}"
        tracks_text = f"ACTIVE: {metrics.active_tracks}"
        fps_text = f"FPS: {metrics.fps:.1f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 32

        # 1. Project Title
        cv2.putText(frame, title_text, (20, y_pos), font, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

        # 2. Left Traffic Flow
        cv2.putText(frame, left_text, (280, y_pos), font, 0.55, (46, 204, 113), 2, cv2.LINE_AA)

        # 3. Right Traffic Flow
        cv2.putText(frame, right_text, (520, y_pos), font, 0.55, (255, 150, 50), 2, cv2.LINE_AA)

        # 4. Active Tracks
        cv2.putText(frame, tracks_text, (760, y_pos), font, 0.55, (200, 200, 255), 2, cv2.LINE_AA)

        # 5. Pipeline FPS
        cv2.putText(frame, fps_text, (w - 150, y_pos), font, 0.60, (0, 255, 127), 2, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        vehicles: Sequence[TrackedVehicle],
        metrics: PipelineMetrics,
        lane_speeds: Sequence[float],
        active_gate_triggers: Sequence[str] = (),
    ) -> np.ndarray:
        """
        Orchestrate all visualization layers onto the frame.

        Args:
            frame: Raw BGR frame.
            vehicles: List of tracked vehicles.
            metrics: Current pipeline metrics.
            lane_speeds: Speeds per lane.
            active_gate_triggers: Gates triggered in current frame.

        Returns:
            Annotated frame ready for display or output encoding.
        """
        out_frame = frame.copy()

        if self.show_lanes:
            self.draw_lanes(out_frame)
        if self.show_lane_speeds:
            self.draw_lane_speeds(out_frame, lane_speeds)
        if self.show_counting_lines:
            self.draw_counting_gates(out_frame, active_gate_triggers)
        if self.show_bounding_boxes:
            self.draw_tracked_vehicles(out_frame, vehicles)
        if self.show_hud_header:
            self.draw_hud_header(out_frame, metrics)

        return out_frame
