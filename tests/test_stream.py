"""Tests for core/stream.py — CameraBuffer and CameraStream.

TDD order: unhappy/failing paths first, then happy paths.
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from core.stream import CameraBuffer, CameraStream


# ---------------------------------------------------------------------------
# CameraBuffer — unhappy paths first
# ---------------------------------------------------------------------------

class TestCameraBufferUnhappy:
    def test_get_frame_returns_none_when_empty(self):
        buf = CameraBuffer()
        assert buf.get_frame() is None

    def test_get_frame_returns_none_after_init_no_put(self):
        buf = CameraBuffer()
        # Multiple reads before any put should all return None
        assert buf.get_frame() is None
        assert buf.get_frame() is None


class TestCameraBufferHappy:
    def test_put_and_get_frame(self):
        buf = CameraBuffer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        buf.put_frame(frame)
        result = buf.get_frame()
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_get_frame_returns_latest(self):
        buf = CameraBuffer()
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 255
        buf.put_frame(frame1)
        buf.put_frame(frame2)
        result = buf.get_frame()
        assert np.array_equal(result, frame2)

    def test_thread_safe_put_and_get(self):
        buf = CameraBuffer()
        results = []

        def writer():
            for _ in range(50):
                buf.put_frame(np.zeros((100, 100, 3), dtype=np.uint8))

        def reader():
            for _ in range(50):
                results.append(buf.get_frame())

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(); t2.join()
        # Should not raise — no assertion on values, just no exception


# ---------------------------------------------------------------------------
# CameraStream — unhappy paths first
# ---------------------------------------------------------------------------

class TestCameraStreamUnhappy:
    def test_invalid_url_does_not_raise_on_init(self):
        # Construction itself must never raise regardless of URL
        stream = CameraStream("cam1", "Test", "http://nonexistent.invalid/video")
        assert stream is not None

    @patch("core.stream.cv2.VideoCapture")
    def test_failed_open_triggers_retry(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        stream = CameraStream("cam1", "Test", "http://bad.url/video")
        stream._max_retries = 2
        stream._retry_base_s = 0.01  # fast for tests

        with patch("core.stream.time.sleep"):
            stream._connect_with_backoff()

        assert mock_cap_cls.call_count >= 2

    @patch("core.stream.cv2.VideoCapture")
    def test_read_failure_does_not_crash_thread(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # every read fails
        mock_cap_cls.return_value = mock_cap

        stream = CameraStream("cam1", "Test", "0")
        stream._max_retries = 1
        stream._retry_base_s = 0.01

        with patch("core.stream.time.sleep"):
            # Run one iteration of the capture loop — should not raise
            stream._run_once()


# ---------------------------------------------------------------------------
# CameraStream — happy paths
# ---------------------------------------------------------------------------

class TestCameraStreamHappy:
    def test_attributes_set_on_init(self):
        stream = CameraStream("cam1", "Drone-1", "http://192.168.1.10:8080/video")
        assert stream.camera_id == "cam1"
        assert stream.name == "Drone-1"
        assert stream.url == "http://192.168.1.10:8080/video"

    def test_buffer_is_camera_buffer(self):
        stream = CameraStream("cam1", "Drone-1", "0")
        assert isinstance(stream.buffer, CameraBuffer)

    @patch("core.stream.cv2.VideoCapture")
    def test_successful_read_stores_frame_in_buffer(self, mock_cap_cls):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        stream = CameraStream("cam1", "Test", "0")
        stream._run_once()

        result = stream.buffer.get_frame()
        assert result is not None
        assert result.shape == frame.shape

    def test_start_creates_and_starts_daemon_thread(self):
        stream = CameraStream("cam1", "Test", "0")
        with patch.object(stream, "_capture_loop"):
            stream.start()
            assert stream._thread is not None
            assert stream._thread.daemon is True

    def test_stop_signals_thread_to_exit(self):
        stream = CameraStream("cam1", "Test", "0")
        stream.start()
        stream.stop()
        stream._thread.join(timeout=2)
        assert not stream._thread.is_alive()
