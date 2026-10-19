"""
Utility helpers for geometry, video I/O, device management, and high-performance drawing.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from src.schemas import BBox, IntPoint, Point


def select_optimal_device(requested_device: str = "auto") -> str:
    """Select the best available hardware accelerator (CUDA, MPS, or CPU)."""
    if requested_device.lower() != "auto":
        return requested_device.lower()

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def point_in_polygon(point: Point | IntPoint, polygon: List[IntPoint] | np.ndarray) -> bool:
    """Determine whether a 2D point lies within a polygon using ray casting."""
    poly_np = np.asarray(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(poly_np, (float(point[0]), float(point[1])), False)
    return result >= 0


def euclidean_distance(p1: Point | IntPoint, p2: Point | IntPoint) -> float:
    """Compute standard 2D Euclidean distance between two points."""
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    box_a_area = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    box_b_area = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_and_prep_mask(
    mask_path: Optional[str], target_width: int, target_height: int
) -> Optional[np.ndarray]:
    """Load and resize an ROI mask to match the target frame dimensions."""
    if not mask_path or not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path)
    if mask is None:
        return None

    if mask.shape[1] != target_width or mask.shape[0] != target_height:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return mask


def draw_styled_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    bg_color: Tuple[int, int, int] = (20, 20, 20),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.45,
    thickness: int = 1,
    padding: int = 4,
    border_radius: int = 3,
) -> None:
    """Render a clean pill badge text label with a solid background and padding."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = origin
    x1 = max(0, x)
    y1 = max(0, y - text_h - padding * 2)
    x2 = min(image.shape[1] - 1, x + text_w + padding * 2)
    y2 = min(image.shape[0] - 1, y)

    # Background rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    # Border
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # Text
    text_x = x1 + padding
    text_y = y2 - padding - baseline // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
