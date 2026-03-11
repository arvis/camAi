"""core/detector.py — YOLOv8 object detection + ByteTrack tracking with OpenCV overlay.

Detector wraps ultralytics.YOLO, runs inference every N frames (per camera),
filters to target classes, draws bounding boxes, tracks objects across frames,
and returns annotated frame, per-class counts, and per-object tracking data.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

TARGET_CLASSES = {
    "person", "car", "truck", "motorcycle", "bicycle",
    # bags (for abandoned-object detection)
    "backpack", "handbag", "suitcase",
    # weapons
    "knife",
}

_CENTROID_HISTORY = 10
_STATIONARY_THRESHOLD = 15

# BGR overlay colours
_COLOR_DEFAULT = (0, 255, 0)    # green — people, vehicles
_COLOR_WEAPON = (0, 0, 255)     # red   — knife, pistol, rifle, assault_rifle
_COLOR_BAG = (0, 215, 255)      # yellow — backpack, handbag, suitcase

_BAG_CLASSES = {"backpack", "handbag", "suitcase"}
_WEAPON_CLASSES = {"knife", "pistol", "rifle", "assault_rifle"}


def _box_color(class_name: str) -> tuple[int, int, int]:
    if class_name in _WEAPON_CLASSES:
        return _COLOR_WEAPON
    if class_name in _BAG_CLASSES:
        return _COLOR_BAG
    return _COLOR_DEFAULT


def _compute_direction(centroids: deque) -> str:
    """Return compass direction from oldest to newest centroid, or 'stationary'."""
    if len(centroids) < 2:
        return "stationary"
    ox, oy = centroids[0]
    nx, ny = centroids[-1]
    dx, dy = nx - ox, ny - oy
    if math.hypot(dx, dy) < _STATIONARY_THRESHOLD:
        return "stationary"
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


class Detector:
    """Run YOLOv8 detection + ByteTrack on camera frames.

    Parameters
    ----------
    model_path : str
        Path or name of the YOLO weights file (auto-downloaded if needed).
    interval_frames : int
        Run detection every N frames per camera; return last results otherwise.
    weapon_model_path : str | None
        Optional path to a second YOLO model trained on pistol/rifle/assault_rifle.
        If None, firearm detection is silently skipped.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        interval_frames: int = 3,
        weapon_model_path: str | None = None,
    ) -> None:
        self._model = YOLO(model_path)
        self._weapon_model = YOLO(weapon_model_path) if weapon_model_path else None
        self._interval = interval_frames
        self._frame_counts: dict[str, int] = {}
        self._last_counts: dict[str, dict[str, int]] = {}
        self._last_tracked: dict[str, list[dict]] = {}
        self._centroids: dict[int, deque] = {}  # track_id → centroid history
        self._last_centroids: dict[int, tuple[float, float]] = {}  # for velocity_px

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, camera_id: str, frame: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int], list[dict]]:
        """Process one frame from a camera.

        Returns
        -------
        annotated_frame : np.ndarray
            Copy of *frame* with bounding boxes drawn (or original if skipped).
        counts : dict[str, int]
            Per-class detection counts, e.g. ``{"person": 2, "car": 1}``.
            Returns the most recent counts on skipped frames.
        tracked_objects : list[dict]
            Per-object tracking data with keys: track_id, class_name,
            movement_direction, bbox, confidence, velocity_px.
            Returns the most recent list on skipped frames.
        """
        count = self._frame_counts.get(camera_id, 0)
        self._frame_counts[camera_id] = count + 1

        if count % self._interval != 0:
            return (
                frame,
                self._last_counts.get(camera_id, {}),
                self._last_tracked.get(camera_id, []),
            )

        return self._detect(camera_id, frame)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect(
        self, camera_id: str, frame: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int], list[dict]]:
        annotated = frame.copy()
        counts: dict[str, int] = {}
        tracked: list[dict] = []

        self._run_model(self._model, TARGET_CLASSES, frame, annotated, counts, tracked)

        if self._weapon_model is not None:
            # Weapon model may detect classes not in TARGET_CLASSES (pistol, rifle, etc.)
            weapon_classes = set(self._weapon_model.names.values())
            self._run_model(self._weapon_model, weapon_classes, frame, annotated, counts, tracked)

        self._last_counts[camera_id] = counts
        self._last_tracked[camera_id] = tracked
        return annotated, counts, tracked

    def _run_model(
        self,
        model,
        allowed_classes: set[str],
        frame: np.ndarray,
        annotated: np.ndarray,
        counts: dict[str, int],
        tracked: list[dict],
    ) -> None:
        results = model.track(
            frame, tracker="bytetrack.yaml", persist=True, verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                if cls_name not in allowed_classes:
                    continue

                counts[cls_name] = counts.get(cls_name, 0) + 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                velocity_px = 0.0
                if track_id is not None:
                    if track_id not in self._centroids:
                        self._centroids[track_id] = deque(maxlen=_CENTROID_HISTORY)
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if track_id in self._last_centroids:
                        prev_cx, prev_cy = self._last_centroids[track_id]
                        velocity_px = math.hypot(cx - prev_cx, cy - prev_cy)
                    self._last_centroids[track_id] = (cx, cy)
                    self._centroids[track_id].append((int(cx), int(cy)))
                    direction = _compute_direction(self._centroids[track_id])
                else:
                    direction = "unknown"

                color = _box_color(cls_name)
                id_prefix = f"#{track_id} " if track_id is not None else ""
                label = f"{id_prefix}{cls_name} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                tracked.append({
                    "track_id": track_id,
                    "class_name": cls_name,
                    "movement_direction": direction,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "velocity_px": velocity_px,
                })
