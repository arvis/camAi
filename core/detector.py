"""core/detector.py — YOLOv8 object detection with OpenCV overlay.

Detector wraps ultralytics.YOLO, runs inference every N frames (per camera),
filters to target classes, draws bounding boxes, and returns annotated frame
plus per-class counts.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

TARGET_CLASSES = {"person", "car", "truck", "motorcycle", "bicycle"}

# BGR colours for overlay
_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 255, 0)


class Detector:
    """Run YOLOv8 detection on camera frames.

    Parameters
    ----------
    model_path : str
        Path or name of the YOLO weights file (auto-downloaded if needed).
    interval_frames : int
        Run detection every N frames per camera; return last counts otherwise.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        interval_frames: int = 3,
    ) -> None:
        self._model = YOLO(model_path)
        self._interval = interval_frames
        self._frame_counts: dict[str, int] = {}   # per camera_id
        self._last_counts: dict[str, dict[str, int]] = {}  # per camera_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, camera_id: str, frame: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Process one frame from a camera.

        Returns
        -------
        annotated_frame : np.ndarray
            Copy of *frame* with bounding boxes drawn (or original if skipped).
        counts : dict[str, int]
            Per-class detection counts, e.g. ``{"person": 2, "car": 1}``.
            Returns the most recent counts on skipped frames.
        """
        count = self._frame_counts.get(camera_id, 0)
        self._frame_counts[camera_id] = count + 1

        if count % self._interval != 0:
            return frame, self._last_counts.get(camera_id, {})

        return self._detect(camera_id, frame)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect(
        self, camera_id: str, frame: np.ndarray
    ) -> tuple[np.ndarray, dict[str, int]]:
        results = self._model(frame, verbose=False)
        annotated = frame.copy()
        counts: dict[str, int] = {}

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
                label = f"{cls_name} {conf:.2f}"

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

        self._last_counts[camera_id] = counts
        return annotated, counts
