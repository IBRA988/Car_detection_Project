"""
SORT (Simple Online and Realtime Tracking) implementation with Kalman Filtering.
Tracks detected vehicles across frames, maintaining trajectory history, class labels, and unique IDs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.schemas import BBox, Detection, TrackedVehicle

logger = logging.getLogger(__name__)


def convert_bbox_to_z(bbox: BBox | np.ndarray) -> np.ndarray:
    """
    Convert bounding box [x1, y1, x2, y2] to observation state [cx, cy, s, r]
    where cx, cy is box center, s is area (scale), and r is aspect ratio (w / h).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray) -> BBox:
    """
    Convert state vector [cx, cy, s, r] to bounding box [x1, y1, x2, y2].
    Compatible with NumPy 2.0+ scalar casting.
    """
    arr = np.asarray(x, dtype=float).flatten()
    s = max(0.0, float(arr[2]))
    r = max(1e-4, float(arr[3]))
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0
    cx = float(arr[0])
    cy = float(arr[1])
    return (
        int(cx - w / 2.0),
        int(cy - h / 2.0),
        int(cx + w / 2.0),
        int(cy + h / 2.0),
    )


class KalmanBoxTracker:
    """Represents the internal state of an individual tracked object using a Kalman filter."""

    _count = 0

    def __init__(self, bbox: BBox, class_id: int = 2, class_name: str = "car", confidence: float = 1.0) -> None:
        """Initialize tracker with initial bounding box and metadata."""
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # Constant velocity state transition matrix F
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high initial uncertainty for unobserved velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.history: List[BBox] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

    @classmethod
    def reset_counter(cls) -> None:
        """Reset track ID counter (useful for unit testing)."""
        cls._count = 0

    def update(self, bbox: BBox, confidence: float = 1.0, class_id: Optional[int] = None, class_name: Optional[str] = None) -> None:
        """Update tracker state with a new matched observation."""
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        if class_id is not None:
            self.class_id = class_id
        if class_name is not None:
            self.class_name = class_name
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self) -> BBox:
        """Advance the Kalman filter state and return the predicted bounding box."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = convert_x_to_bbox(self.kf.x)
        self.history.append(bbox)
        return bbox

    def get_state(self) -> BBox:
        """Return the current bounding box estimate from the state vector."""
        return convert_x_to_bbox(self.kf.x)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two arrays of bounding boxes [N, 4] and [M, 4]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=float)

    boxes_a = np.expand_dims(boxes_a, 1)  # (N, 1, 4)
    boxes_b = np.expand_dims(boxes_b, 0)  # (1, M, 4)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])

    union = area_a + area_b - intersection
    union = np.maximum(union, 1e-6)
    return intersection / union


class SortTracker:
    """
    Multi-object tracker based on SORT.
    Matches detections to Kalman trackers using Hungarian matching on IoU distances.
    """

    def __init__(
        self,
        max_age: int = 25,
        min_hits: int = 3,
        iou_threshold: float = 0.30,
        track_history_length: int = 30,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_history_length = track_history_length

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0
        # Preserves centroid trajectories indexed by track ID
        self.trajectories: dict[int, List[Tuple[int, int]]] = {}

    def update(self, detections: Sequence[Detection]) -> List[TrackedVehicle]:
        """
        Update tracker with new detections for the current frame.

        Args:
            detections: List of current frame detections.

        Returns:
            List of confirmed TrackedVehicle objects.
        """
        self.frame_count += 1

        # 1. Predict positions of existing trackers
        predicted_boxes = []
        to_delete = []
        for i, tracker in enumerate(self.trackers):
            pred_box = tracker.predict()
            if any(np.isnan(pred_box)):
                to_delete.append(i)
            else:
                predicted_boxes.append(pred_box)

        for i in reversed(to_delete):
            self.trackers.pop(i)

        det_boxes = np.array([d.bbox for d in detections], dtype=float) if detections else np.empty((0, 4))
        trk_boxes = np.array(predicted_boxes, dtype=float) if predicted_boxes else np.empty((0, 4))

        # 2. Associate detections to trackers using Hungarian matching on IoU
        matched_indices, unmatched_dets, unmatched_trks = self._associate(det_boxes, trk_boxes)

        # 3. Update matched trackers
        for d_idx, t_idx in matched_indices:
            det = detections[d_idx]
            self.trackers[t_idx].update(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
            )
            self.trackers.append(new_trk)

        # 5. Extract active, confirmed tracks and cull expired ones
        active_vehicles: List[TrackedVehicle] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                # Clean up trajectory history when tracker dies
                self.trajectories.pop(trk.id, None)
                continue

            surviving_trackers.append(trk)

            # Return tracks that have satisfied min_hits or early frames
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                current_box = trk.get_state()
                cx = (current_box[0] + current_box[2]) // 2
                cy = (current_box[1] + current_box[3]) // 2

                if trk.id not in self.trajectories:
                    self.trajectories[trk.id] = []
                self.trajectories[trk.id].append((cx, cy))

                # Truncate history
                if len(self.trajectories[trk.id]) > self.track_history_length:
                    self.trajectories[trk.id].pop(0)

                vehicle = TrackedVehicle(
                    track_id=trk.id,
                    bbox=current_box,
                    confidence=trk.confidence,
                    class_id=trk.class_id,
                    class_name=trk.class_name,
                    history=list(self.trajectories[trk.id]),
                )
                active_vehicles.append(vehicle)

        self.trackers = surviving_trackers
        return active_vehicles

    def _associate(
        self, det_boxes: np.ndarray, trk_boxes: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian matching between detections and predicted track boxes based on IoU."""
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []

        iou_matrix = compute_iou_matrix(det_boxes, trk_boxes)

        # linear_sum_assignment finds minimum cost, so cost = -iou
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_dets = set(range(len(det_boxes)))
        unmatched_trks = set(range(len(trk_boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                unmatched_dets.discard(r)
                unmatched_trks.discard(c)

        return matched, sorted(unmatched_dets), sorted(unmatched_trks)
