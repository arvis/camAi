"""api/mjpeg.py — Thread-safe annotated frame store and async MJPEG generator.

AnnotatedFrameStore holds the latest annotated frame per camera (written by
camera processing threads, read by the MJPEG stream endpoint).

mjpeg_stream is an async generator that yields multipart/x-mixed-replace
chunks suitable for StreamingResponse.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

import cv2
import numpy as np


class AnnotatedFrameStore:
    """Thread-safe store for the most recent annotated frame per camera.

    Camera processing threads call put(); the MJPEG endpoint calls get().
    """

    def __init__(self) -> None:
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def put(self, cam_id: str, frame: np.ndarray) -> None:
        """Store the latest annotated frame for cam_id."""
        with self._lock:
            self._frames[cam_id] = frame

    def get(self, cam_id: str) -> np.ndarray | None:
        """Return the most recent frame for cam_id, or None if not yet available."""
        with self._lock:
            return self._frames.get(cam_id)

    def camera_ids(self) -> list[str]:
        """Return all registered camera IDs."""
        with self._lock:
            return list(self._frames.keys())


async def mjpeg_stream(
    frame_store: AnnotatedFrameStore,
    cam_id: str,
    *,
    max_frames: int | None = None,
    sleep_s: float = 0.033,
) -> AsyncIterator[bytes]:
    """Async generator yielding MJPEG multipart chunks for cam_id.

    Parameters
    ----------
    frame_store : AnnotatedFrameStore
    cam_id : str
    max_frames : int | None
        Stop after this many frames (used in tests). None = run forever.
    sleep_s : float
        Seconds to sleep between frames (~30 fps at default 0.033).
    """
    yielded = 0
    while max_frames is None or yielded < max_frames:
        frame = frame_store.get(cam_id)
        if frame is not None:
            ok, jpeg = cv2.imencode(".jpg", frame)
            if ok:
                data = jpeg.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + data
                    + b"\r\n"
                )
                yielded += 1
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        elif frame is None:
            # No frame and no sleep — exit immediately (used in empty-store tests)
            break
