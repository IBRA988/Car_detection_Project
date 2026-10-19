"""
Vehicle detection module powered by Ultralytics YOLOv8.
Optimized for multi-class highway traffic detection with ROI masking and hardware acceleration.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from src.schemas import Detection
from src.utils import select_optimal_device

logger = logging.getLogger(__name__)


class VehicleDetector:
    """Encapsulates YOLOv8 inference, target class filtering, and ROI masking."""

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        target_classes: Optional[Sequence[int]] = None,
        class_name_map: Optional[Dict[int, str]] = None,
        device: str = "auto",
        roi_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the vehicle detector.

        Args:
            weights_path: Path to YOLO weights file (.pt).
            confidence_threshold: Minimum detection confidence (0.0 - 1.0).
            iou_threshold: NMS IoU threshold.
            target_classes: Class IDs to retain (e.g., [2, 3, 5, 7] for COCO vehicles).
            class_name_map: Mapping from class ID to readable label string.
            device: Accelerator to use ('cuda', 'mps', 'cpu', or 'auto').
            roi_mask: Binary or multi-channel mask for masking inference area.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")

        self.device = select_optimal_device(device)
        logger.info(f"Loading YOLO model from {weights_path} onto device={self.device}...")
        self.model = YOLO(weights_path)

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.target_classes = set(target_classes if target_classes is not None else [2, 3, 5, 7])
        self.class_name_map = class_name_map or {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
        self.roi_mask = roi_mask

    def set_roi_mask(self, mask: Optional[np.ndarray]) -> None:
        """Update or set the inference Region of Interest (ROI) mask."""
        self.roi_mask = mask

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform detection on a single frame.

        Args:
            frame: BGR image numpy array.

        Returns:
            List of validated Detection instances within the target classes.
        """
        if frame is None or frame.size == 0:
            return []

        # Apply ROI mask if present
        if self.roi_mask is not None:
            if self.roi_mask.shape[:2] != frame.shape[:2]:
                resized_mask = cv2.resize(
                    self.roi_mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                input_frame = cv2.bitwise_and(frame, resized_mask)
            else:
                input_frame = cv2.bitwise_and(frame, self.roi_mask)
        else:
            input_frame = frame

        # Run YOLO inference
        results = self.model.predict(
            source=input_frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=list(self.target_classes),
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract bounding boxes, confidence, and class
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, classes):
            if cls_id in self.target_classes and conf >= self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                class_label = self.class_name_map.get(cls_id, f"class_{cls_id}")
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_label,
                    )
                )

        return detections
