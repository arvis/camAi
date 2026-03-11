"""Tests for core/detector.py — Detector class.

TDD order: unhappy/failing paths first, then happy paths.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box(cls_id: int, conf: float, x1: int, y1: int, x2: int, y2: int,
              track_id=None):
    """Build a mock YOLO box with list-indexable attributes."""
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [np.array([x1, y1, x2, y2], dtype=float)]
    box.id = [track_id] if track_id is not None else None
    return box


def _make_result(boxes):
    """Build a mock YOLO result with an iterable boxes attribute."""
    result = MagicMock()
    result.boxes = boxes  # list of mock boxes (or empty list)
    return result


def _make_model(names: dict, results_per_call=None):
    """Build a mock YOLO model instance (mocks model.track)."""
    model = MagicMock()
    model.names = names
    results = results_per_call if results_per_call is not None else []
    model.track.return_value = results
    return model


# Common YOLO class names used in tests
NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
         15: "cat", 16: "dog"}


# ---------------------------------------------------------------------------
# Unhappy paths — detection
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
        _, counts, _ = det.process("cam1", frame)

        assert "cat" not in counts
        assert counts == {}

    @patch("core.detector.YOLO")
    def test_empty_boxes_returns_empty_counts(self, mock_yolo_cls):
        """No detections → empty counts dict."""
        from core.detector import Detector

        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts, _ = det.process("cam1", frame)

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
        annotated, counts, _ = det.process("cam1", frame)

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

        assert mock_model.track.call_count == 1

    @patch("core.detector.YOLO")
    def test_skipped_frames_return_last_counts(self, mock_yolo_cls):
        """Skipped frames should return the most recent detection counts."""
        from core.detector import Detector

        person_box = _make_box(cls_id=0, conf=0.85, x1=10, y1=10, x2=100, y2=200)
        mock_model = _make_model(NAMES, [_make_result([person_box])])
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, counts_first, _ = det.process("cam1", frame)   # detects: {"person": 1}
        _, counts_second, _ = det.process("cam1", frame)  # skipped
        _, counts_third, _ = det.process("cam1", frame)   # skipped

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
        _, counts, _ = det.process("cam1", frame)

        assert "bus" not in counts


# ---------------------------------------------------------------------------
# Happy paths — detection
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
        _, counts, _ = det.process("cam1", frame)

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
        _, counts, _ = det.process("cam1", frame)

        assert counts == {"person": 2, "car": 1}

    @patch("core.detector.YOLO")
    def test_annotated_frame_same_shape(self, mock_yolo_cls):
        """Returned annotated frame must have same shape as input."""
        from core.detector import Detector

        person_box = _make_box(cls_id=0, conf=0.9, x1=10, y1=10, x2=100, y2=200)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([person_box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, _, _ = det.process("cam1", frame)

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
        assert mock_model.track.call_count == 3

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
        assert mock_model.track.call_count == 2

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

        assert mock_model.track.call_count == 5

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
        _, counts, _ = det.process("cam1", frame)

        assert set(counts.keys()) == {"person", "car", "motorcycle", "bicycle", "truck"}


# ---------------------------------------------------------------------------
# Unhappy paths — tracking
# ---------------------------------------------------------------------------

class TestDetectorTrackingUnhappy:
    @patch("core.detector.YOLO")
    def test_box_with_no_track_id_returns_unknown_direction(self, mock_yolo_cls):
        """box.id is None → movement_direction is 'unknown'."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.9, x1=10, y1=10, x2=100, y2=200,
                        track_id=None)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert len(tracked) == 1
        assert tracked[0]["track_id"] is None
        assert tracked[0]["movement_direction"] == "unknown"

    @patch("core.detector.YOLO")
    def test_no_detections_returns_empty_tracked_list(self, mock_yolo_cls):
        """Empty boxes → tracked_objects is []."""
        from core.detector import Detector

        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert tracked == []

    @patch("core.detector.YOLO")
    def test_none_boxes_returns_empty_tracked_list(self, mock_yolo_cls):
        """result.boxes is None → tracked_objects is []."""
        from core.detector import Detector

        result = MagicMock()
        result.boxes = None
        mock_yolo_cls.return_value = _make_model(NAMES, [result])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert tracked == []

    @patch("core.detector.YOLO")
    def test_single_frame_with_track_id_returns_stationary(self, mock_yolo_cls):
        """Only one centroid stored → cannot compute direction → stationary."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                        track_id=1)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["movement_direction"] == "stationary"

    @patch("core.detector.YOLO")
    def test_tiny_movement_returns_stationary(self, mock_yolo_cls):
        """Movement of < 15px total → stationary."""
        from core.detector import Detector

        # centroid frame 1: (150, 150)
        box1 = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                         track_id=1)
        # centroid frame 2: (155, 153) — movement of ~5.8px
        box2 = _make_box(cls_id=0, conf=0.9, x1=105, y1=103, x2=205, y2=203,
                         track_id=1)

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.side_effect = [
            [_make_result([box1])],
            [_make_result([box2])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["movement_direction"] == "stationary"

    @patch("core.detector.YOLO")
    def test_non_target_class_not_in_tracked_list(self, mock_yolo_cls):
        """Non-target class (cat) must not appear in tracked_objects."""
        from core.detector import Detector

        cat_box = _make_box(cls_id=15, conf=0.9, x1=10, y1=10, x2=50, y2=50,
                            track_id=5)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([cat_box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert tracked == []

    @patch("core.detector.YOLO")
    def test_skipped_frame_returns_last_tracked(self, mock_yolo_cls):
        """Skipped frames return cached tracked_objects from last detection."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.9, x1=10, y1=10, x2=100, y2=200,
                        track_id=7)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([box])])

        det = Detector(interval_frames=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        _, _, tracked_first = det.process("cam1", frame)   # frame 0 → detects
        _, _, tracked_skip = det.process("cam1", frame)    # frame 1 → skipped

        assert len(tracked_first) == 1
        assert tracked_skip == tracked_first


# ---------------------------------------------------------------------------
# Happy paths — tracking
# ---------------------------------------------------------------------------

class TestDetectorTrackingHappy:
    @patch("core.detector.YOLO")
    def test_box_with_track_id_returned(self, mock_yolo_cls):
        """Valid track_id is present in tracked_objects."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.9, x1=10, y1=10, x2=100, y2=200,
                        track_id=42)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert len(tracked) == 1
        assert tracked[0]["track_id"] == 42

    @patch("core.detector.YOLO")
    def test_tracked_object_has_required_fields(self, mock_yolo_cls):
        """tracked_objects entries must have track_id, class_name, movement_direction, bbox, confidence."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.85, x1=10, y1=10, x2=100, y2=200,
                        track_id=1)
        mock_yolo_cls.return_value = _make_model(NAMES, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        obj = tracked[0]
        assert "track_id" in obj
        assert "class_name" in obj
        assert "movement_direction" in obj
        assert "bbox" in obj
        assert "confidence" in obj
        assert obj["class_name"] == "person"
        assert abs(obj["confidence"] - 0.85) < 1e-4

    @patch("core.detector.YOLO")
    def test_movement_south(self, mock_yolo_cls):
        """Centroid moves down (dy > 0, |dy| > |dx|) → direction S."""
        from core.detector import Detector

        # centroid: (150, 150)
        box1 = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                         track_id=1)
        # centroid: (150, 300) — dy=150, dx=0 → S
        box2 = _make_box(cls_id=0, conf=0.9, x1=100, y1=250, x2=200, y2=350,
                         track_id=1)

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.side_effect = [
            [_make_result([box1])],
            [_make_result([box2])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["movement_direction"] == "S"

    @patch("core.detector.YOLO")
    def test_movement_north(self, mock_yolo_cls):
        """Centroid moves up (dy < 0, |dy| > |dx|) → direction N."""
        from core.detector import Detector

        box1 = _make_box(cls_id=0, conf=0.9, x1=100, y1=250, x2=200, y2=350,
                         track_id=2)
        box2 = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                         track_id=2)

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.side_effect = [
            [_make_result([box1])],
            [_make_result([box2])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["movement_direction"] == "N"

    @patch("core.detector.YOLO")
    def test_movement_east(self, mock_yolo_cls):
        """Centroid moves right (dx > 0, |dx| >= |dy|) → direction E."""
        from core.detector import Detector

        box1 = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                         track_id=3)
        box2 = _make_box(cls_id=0, conf=0.9, x1=300, y1=100, x2=400, y2=200,
                         track_id=3)

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.side_effect = [
            [_make_result([box1])],
            [_make_result([box2])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["movement_direction"] == "E"

    @patch("core.detector.YOLO")
    def test_movement_west(self, mock_yolo_cls):
        """Centroid moves left (dx < 0, |dx| >= |dy|) → direction W."""
        from core.detector import Detector

        box1 = _make_box(cls_id=0, conf=0.9, x1=300, y1=100, x2=400, y2=200,
                         track_id=4)
        box2 = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                         track_id=4)

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.side_effect = [
            [_make_result([box1])],
            [_make_result([box2])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["movement_direction"] == "W"

    @patch("core.detector.YOLO")
    def test_multiple_tracks_independent(self, mock_yolo_cls):
        """Two different track IDs maintain independent centroid histories."""
        from core.detector import Detector

        # Track 1 moves east, track 2 moves south
        b1a = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200, track_id=1)
        b2a = _make_box(cls_id=0, conf=0.9, x1=300, y1=100, x2=400, y2=200, track_id=2)

        b1b = _make_box(cls_id=0, conf=0.9, x1=300, y1=100, x2=400, y2=200, track_id=1)
        b2b = _make_box(cls_id=0, conf=0.9, x1=300, y1=300, x2=400, y2=400, track_id=2)

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.side_effect = [
            [_make_result([b1a, b2a])],
            [_make_result([b1b, b2b])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        by_id = {obj["track_id"]: obj for obj in tracked}
        assert by_id[1]["movement_direction"] == "E"
        assert by_id[2]["movement_direction"] == "S"

    @patch("core.detector.YOLO")
    def test_centroid_history_capped_at_ten(self, mock_yolo_cls):
        """Centroid history is capped at 10 — oldest entry dropped after 10 frames."""
        from core.detector import Detector

        # All boxes at same position — stationary throughout
        boxes = [
            _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200,
                      track_id=1)
        ]

        mock_model = MagicMock()
        mock_model.names = NAMES
        mock_model.track.return_value = [_make_result(boxes)]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(15):
            det.process("cam1", frame)

        # Internal centroid deque must not grow beyond 10
        assert len(det._centroids[1]) <= 10


# ---------------------------------------------------------------------------
# Step 4 additions — new TARGET_CLASSES, velocity_px, weapon model, colours
# ---------------------------------------------------------------------------

# Extended names dict that includes the new classes
NAMES_EXT = {
    0: "person", 2: "car", 7: "truck", 3: "motorcycle", 1: "bicycle",
    24: "backpack", 26: "handbag", 28: "suitcase", 43: "knife",
}


class TestDetectorStep4UnhappyClasses:

    @patch("core.detector.YOLO")
    def test_new_bag_classes_included(self, mock_yolo_cls):
        """backpack, handbag, suitcase must appear in counts."""
        from core.detector import Detector

        boxes = [
            _make_box(cls_id=24, conf=0.8, x1=0, y1=0, x2=50, y2=50),  # backpack
            _make_box(cls_id=26, conf=0.8, x1=60, y1=0, x2=110, y2=50),  # handbag
            _make_box(cls_id=28, conf=0.8, x1=120, y1=0, x2=170, y2=50),  # suitcase
        ]
        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result(boxes)])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts, _ = det.process("cam1", frame)

        assert "backpack" in counts
        assert "handbag" in counts
        assert "suitcase" in counts

    @patch("core.detector.YOLO")
    def test_knife_included_in_target_classes(self, mock_yolo_cls):
        """knife must appear in counts."""
        from core.detector import Detector

        box = _make_box(cls_id=43, conf=0.9, x1=0, y1=0, x2=30, y2=30)
        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts, _ = det.process("cam1", frame)

        assert "knife" in counts

    @patch("core.detector.YOLO")
    def test_velocity_px_zero_on_first_appearance(self, mock_yolo_cls):
        """velocity_px is 0.0 when a track is seen for the first time."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200, track_id=1)
        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert tracked[0]["velocity_px"] == 0.0

    @patch("core.detector.YOLO")
    def test_velocity_px_not_present_for_untracked_box(self, mock_yolo_cls):
        """Box with no track_id still has velocity_px = 0.0 in tracked output."""
        from core.detector import Detector

        box = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200, track_id=None)
        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result([box])])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        assert "velocity_px" in tracked[0]
        assert tracked[0]["velocity_px"] == 0.0


class TestDetectorStep4HappyClasses:

    @patch("core.detector.YOLO")
    def test_velocity_px_correct_after_movement(self, mock_yolo_cls):
        """velocity_px equals Euclidean distance between consecutive centroids."""
        from core.detector import Detector

        # centroid 1: (150, 150)
        box1 = _make_box(cls_id=0, conf=0.9, x1=100, y1=100, x2=200, y2=200, track_id=1)
        # centroid 2: (150, 250) — dy=100, dx=0 → velocity=100.0
        box2 = _make_box(cls_id=0, conf=0.9, x1=100, y1=200, x2=200, y2=300, track_id=1)

        mock_model = MagicMock()
        mock_model.names = NAMES_EXT
        mock_model.track.side_effect = [
            [_make_result([box1])],
            [_make_result([box2])],
        ]
        mock_yolo_cls.return_value = mock_model

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det.process("cam1", frame)
        _, _, tracked = det.process("cam1", frame)

        assert abs(tracked[0]["velocity_px"] - 100.0) < 1e-6

    @patch("core.detector.YOLO")
    def test_all_new_target_classes_in_tracked(self, mock_yolo_cls):
        """backpack, handbag, suitcase, knife all appear in tracked_objects."""
        from core.detector import Detector

        boxes = [
            _make_box(cls_id=24, conf=0.8, x1=0, y1=0, x2=50, y2=50, track_id=1),
            _make_box(cls_id=26, conf=0.8, x1=60, y1=0, x2=110, y2=50, track_id=2),
            _make_box(cls_id=28, conf=0.8, x1=120, y1=0, x2=170, y2=50, track_id=3),
            _make_box(cls_id=43, conf=0.9, x1=0, y1=60, x2=30, y2=90, track_id=4),
        ]
        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result(boxes)])

        det = Detector(interval_frames=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        class_names = {obj["class_name"] for obj in tracked}
        assert class_names == {"backpack", "handbag", "suitcase", "knife"}


class TestDetectorStep4WeaponModel:

    @patch("core.detector.YOLO")
    def test_weapon_model_not_loaded_when_path_is_none(self, mock_yolo_cls):
        """No second YOLO instance created when weapon_model_path is None."""
        from core.detector import Detector

        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result([])])
        Detector(interval_frames=1, weapon_model_path=None)
        # YOLO called exactly once (main model only)
        assert mock_yolo_cls.call_count == 1

    @patch("core.detector.YOLO")
    def test_weapon_model_loaded_when_path_provided(self, mock_yolo_cls):
        """Second YOLO instance created when weapon_model_path is given."""
        from core.detector import Detector

        mock_yolo_cls.return_value = _make_model(NAMES_EXT, [_make_result([])])
        Detector(interval_frames=1, weapon_model_path="weapons.pt")
        assert mock_yolo_cls.call_count == 2

    @patch("core.detector.YOLO")
    def test_weapon_model_results_merged_into_counts(self, mock_yolo_cls):
        """Detections from weapon model are merged into the same counts dict."""
        from core.detector import Detector

        weapon_names = {0: "pistol", 1: "rifle"}
        main_model = _make_model(NAMES_EXT, [_make_result([])])
        weapon_model = _make_model(weapon_names, [
            _make_result([_make_box(cls_id=0, conf=0.95, x1=10, y1=10, x2=50, y2=50)])
        ])
        mock_yolo_cls.side_effect = [main_model, weapon_model]

        det = Detector(interval_frames=1, weapon_model_path="weapons.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, counts, _ = det.process("cam1", frame)

        assert "pistol" in counts
        assert counts["pistol"] == 1

    @patch("core.detector.YOLO")
    def test_weapon_model_tracked_objects_merged(self, mock_yolo_cls):
        """Tracked objects from weapon model appear in the tracked list."""
        from core.detector import Detector

        weapon_names = {0: "pistol"}
        main_model = _make_model(NAMES_EXT, [_make_result([])])
        weapon_model = _make_model(weapon_names, [
            _make_result([_make_box(cls_id=0, conf=0.95, x1=10, y1=10, x2=50, y2=50,
                                    track_id=99)])
        ])
        mock_yolo_cls.side_effect = [main_model, weapon_model]

        det = Detector(interval_frames=1, weapon_model_path="weapons.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, _, tracked = det.process("cam1", frame)

        class_names = {obj["class_name"] for obj in tracked}
        assert "pistol" in class_names
