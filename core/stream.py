"""core/stream.py — Camera stream ingestion.

One CameraStream thread per camera source.
Sources can be:
  - MJPEG URL   → "http://192.168.1.10:8080/video"
  - Local file  → "./test_footage/drone.mp4"
  - Webcam      → 0  (integer as string from YAML, or int)
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraBuffer:
    """Thread-safe single-slot frame buffer (latest frame only)."""

    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def put_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame


class CameraStream:
    """Captures frames from a single camera source in a background thread.

    Parameters
    ----------
    camera_id : str
        Logical ID matching config.yaml (e.g. "cam1").
    name : str
        Human-readable label shown in dashboard / alerts.
    url : str | int
        cv2.VideoCapture source — URL, file path, or webcam index.
    """

    # Retry / backoff config (overridable in tests)
    _retry_base_s: float = 5.0
    _retry_max_s: float = 60.0
    _max_retries: int = 0  # 0 = retry forever

    def __init__(self, camera_id: str, name: str, url) -> None:
        self.camera_id = camera_id
        self.name = name
        # YAML may give "0" as a string; cv2 needs int for webcam index
        self.url = int(url) if str(url).lstrip("-").isdigit() else url
        self.buffer = CameraBuffer()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"stream-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to exit and return."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal helpers (also called directly by unit tests)
    # ------------------------------------------------------------------

    def _connect_with_backoff(self) -> Optional[cv2.VideoCapture]:
        """Try to open the source; retry with exponential backoff on failure.

        Returns an open VideoCapture, or None if _max_retries exceeded.
        """
        delay = self._retry_base_s
        attempts = 0
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.url)
            if cap.isOpened():
                return cap
            cap.release()
            attempts += 1
            if self._max_retries and attempts >= self._max_retries:
                return None
            time.sleep(delay)
            delay = min(delay * 2, self._retry_max_s)
        return None

    def _run_once(self) -> bool:
        """Open source, read one frame, store it, release.

        Used by tests to exercise a single read cycle.
        Returns True if a frame was read successfully.
        """
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            cap.release()
            return False
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            self.buffer.put_frame(frame)
            return True
        return False

    def _capture_loop(self) -> None:
        """Main loop: connect, read frames until disconnect, then reconnect."""
        while not self._stop_event.is_set():
            cap = self._connect_with_backoff()
            if cap is None:
                break  # max retries exceeded or stop requested
            try:
                while not self._stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break  # source disconnected — reconnect
                    self.buffer.put_frame(frame)
            finally:
                cap.release()
