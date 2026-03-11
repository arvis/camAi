"""Tests for core/detector.py — Detector class.

TDD order: unhappy/failing paths first, then happy paths.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box(cls_id: int, conf: float, x1: int, y1: int, x2: int, y2: int):
    """Build a mock YOLO box with list-indexable attributes."""
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [np.array([x1, y1, x2, y2], dtype=float)]
    return box


def _make_result(boxes):
    """Build a mock YOLO result with an iterable boxes attribute."""
    result = MagicMock()
    result.boxes = boxes  # list of mock boxes (or empty list)
    return result


def _make_model(names: dict, results_per_call=None):
    """Build a mock YOLO model instance."""
    model = MagicMock()
    model.names = names
    if results_per_call is not None:
        model.return_value = results_per_call
    else:
        model.return_value = []
    return model


# Common YOLO class names used in tests
NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
         15: "cat", 16: "dog"}


# ---------------------------------------------------------------------------
# Unhappy paths
# ---------------------------------------------------------------------------

class TestDetectorUnhappy:
    @patch("core.detector.YOLO")
    def test_non_target_class_excluded_from_counts(self, mock_yolo_cls):
        """Classes like 'cat' must not appear in counts."""
        from core.detector import Detector

        cat_box = _make_box(cls_id=15, conf=0.9, x1=10, y1=10, x2=50, y2=50)
        mock_yolo_cls.return_value = _make_model(
            NAMES, [_make_result([cat_box])]
        )

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts = det.process("cam1", frame)

        assert "cat" not in counts
        assert counts == {}

    @patch("core.detector.YOLO")
    def test_empty_boxes_returns_empty_counts(self, mock_yolo_cls):
        """No detections → empty counts dict."""
        from core.detector import Detector

        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts = det.process("cam1", frame)

        assert counts == {}

    @patch("core.detector.YOLO")
    def test_none_boxes_does_not_crash(self, mock_yolo_cls):
        """result.boxes = None must not raise."""
        from core.detector import Detector

        result = MagicMock()
        result.boxes = None
        mock_yolo_cls.return_value = _make_model(NAMES, [result])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, counts = det.process("cam1", frame)

        assert counts == {}
        assert annotated.shape == frame.shape

    @patch("core.detector.YOLO")
    def test_skipped_frames_do_not_call_model(self, mock_yolo_cls):
        """Model must not be called on frames between detection intervals."""
        from core.detector import Detector

        mock_model = _make_model(NAMES, [_make_result([])])
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)  # frame 0 → detects
        det.process("cam1", frame)  # frame 1 → skip
        det.process("cam1", frame)  # frame 2 → skip

        assert mock_model.call_count == 1

    @patch("core.detector.YOLO")
    def test_skipped_frames_return_last_counts(self, mock_yolo_cls):
        """Skipped frames should return the most recent detection counts."""
        from core.detector import Detector

        person_box = _make_box(cls_id=0, conf=0.85, x1=10, y1=10, x2=100, y2=200)
        mock_model = _make_model(NAMES, [_make_result([person_box])])
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, counts_first = det.process("cam1", frame)   # detects: {"person": 1}
        _, counts_second = det.process("cam1", frame)  # skipped
        _, counts_third = det.process("cam1", frame)   # skipped

        assert counts_first == {"person": 1}
        assert counts_second == {"person": 1}
        assert counts_third == {"person": 1}

    @patch("core.detector.YOLO")
    def test_bus_excluded_from_counts(self, mock_yolo_cls):
        """'bus' is not in target classes — should not appear in counts."""
        from core.detector import Detector

        bus_box = _make_box(cls_id=5, conf=0.8, x1=0, y1=0, x2=100, y2=100)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([bus_box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts = det.process("cam1", frame)

        assert "bus" not in counts


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestDetectorHappy:
    @patch("core.detector.YOLO")
    def test_single_person_counted(self, mock_yolo_cls):
        """One person box → counts = {'person': 1}."""
        from core.detector import Detector

        person_box = _make_box(cls_id=0, conf=0.9, x1=10, y1=10, x2=100, y2=200)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([person_box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts = det.process("cam1", frame)

        assert counts == {"person": 1}

    @patch("core.detector.YOLO")
    def test_multiple_classes_counted(self, mock_yolo_cls):
        """Two persons + one car → correct counts."""
        from core.detector import Detector

        boxes = [
            _make_box(cls_id=0, conf=0.9, x1=0, y1=0, x2=50, y2=100),
            _make_box(cls_id=0, conf=0.8, x1=60, y1=0, x2=110, y2=100),
            _make_box(cls_id=2, conf=0.7, x1=200, y1=50, x2=400, y2=150),
        ]
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result(boxes)])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts = det.process("cam1", frame)

        assert counts == {"person": 2, "car": 1}

    @patch("core.detector.YOLO")
    def test_annotated_frame_same_shape(self, mock_yolo_cls):
        """Returned annotated frame must have same shape as input."""
        from core.detector import Detector

        person_box = _make_box(cls_id=0, conf=0.9, x1=10, y1=10, x2=100, y2=200)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([person_box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, _ = det.process("cam1", frame)

        assert annotated.shape == frame.shape

    @patch("core.detector.YOLO")
    def test_detection_runs_every_interval(self, mock_yolo_cls):
        """With interval=3, model called on frames 0, 3, 6."""
        from core.detector import Detector

        mock_model = _make_model(NAMES, [_make_result([])])
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(7):  # frames 0–6
            det.process("cam1", frame)

        # frames 0, 3, 6 → 3 calls
        assert mock_model.call_count == 3

    @patch("core.detector.YOLO")
    def test_independent_frame_counters_per_camera(self, mock_yolo_cls):
        """Different camera_ids have independent frame counters."""
        from core.detector import Detector

        mock_model = _make_model(NAMES, [_make_result([])])
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # cam1: frame 0 → detect
        det.process("cam1", frame)
        # cam2: frame 0 → detect (independent counter)
        det.process("cam2", frame)

        # Both first frames → 2 model calls
        assert mock_model.call_count == 2

    @patch("core.detector.YOLO")
    def test_interval_one_detects_every_frame(self, mock_yolo_cls):
        """interval_frames=1 means detection on every single frame."""
        from core.detector import Detector

        mock_model = _make_model(NAMES, [_make_result([])])
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(5):
            det.process("cam1", frame)

        assert mock_model.call_count == 5

    @patch("core.detector.YOLO")
    def test_model_loaded_once_at_init(self, mock_yolo_cls):
        """YOLO constructor called exactly once regardless of frames processed."""
        from core.detector import Detector

        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(10):
            det.process("cam1", frame)

        assert mock_yolo_cls.call_count == 1

    @patch("core.detector.YOLO")
    def test_all_target_classes_accepted(self, mock_yolo_cls):
        """person, car, truck, motorcycle, bicycle all appear in counts."""
        from core.detector import Detector

        names = {0: "person", 2: "car", 3: "motorcycle", 1: "bicycle", 7: "truck"}
        boxes = [_make_box(cls_id=i, conf=0.8, x1=0, y1=0, x2=50, y2=50)
                 for i in names]
        mock_yolo_cls.return_value = _make_model(names, [_make_result(boxes)])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts = det.process("cam1", frame)

        assert set(counts.keys()) == {"person", "car", "motorcycle", "bicycle", "truck"}
