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

TARGET_CLASSES = {"person", "car", "truck", "motorcycle", "bicycle"}

_CENTROID_HISTORY = 10
_STATIONARY_THRESHOLD = 15

# BGR colours for overlay
_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 255, 0)


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
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        interval_frames: int = 3,
    ) -> None:
        self._model = YOLO(model_path)
        self._interval = interval_frames
        self._frame_counts: dict[str, int] = {}
        self._last_counts: dict[str, dict[str, int]] = {}
        self._last_tracked: dict[str, list[dict]] = {}
        self._centroids: dict[int, deque] = {}  # track_id → centroid history

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
            movement_direction, bbox, confidence.
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
        results = self._model.track(
            frame, tracker="bytetrack.yaml", persist=True, verbose=False
        )
        annotated = frame.copy()
        counts: dict[str, int] = {}
        tracked: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self._model.names[cls_id]
                if cls_name not in TARGET_CLASSES:
                    continue

                counts[cls_name] = counts.get(cls_name, 0) + 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Track ID and movement direction
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])

                if track_id is not None:
                    if track_id not in self._centroids:
                        self._centroids[track_id] = deque(maxlen=_CENTROID_HISTORY)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    self._centroids[track_id].append((cx, cy))
                    direction = _compute_direction(self._centroids[track_id])
                else:
                    direction = "unknown"

                # Overlay
                id_prefix = f"#{track_id} " if track_id is not None else ""
                label = f"{id_prefix}{cls_name} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

                tracked.append({
                    "track_id": track_id,
                    "class_name": cls_name,
                    "movement_direction": direction,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })

        self._last_counts[camera_id] = counts
        self._last_tracked[camera_id] = tracked
        return annotated, counts, tracked
