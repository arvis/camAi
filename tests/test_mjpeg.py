"""tests/test_mjpeg.py — Unit tests for AnnotatedFrameStore and MJPEG generator.

TDD: unhappy / edge cases first, then happy paths.
"""

import asyncio

import numpy as np
import pytest

from api.mjpeg import AnnotatedFrameStore, mjpeg_stream


# ---------------------------------------------------------------------------
# AnnotatedFrameStore — unhappy paths
# ---------------------------------------------------------------------------

def test_get_unknown_camera_returns_none():
    store = AnnotatedFrameStore()
    assert store.get("cam99") is None


def test_get_before_any_put_returns_none():
    store = AnnotatedFrameStore()
    assert store.get("cam1") is None


def test_camera_ids_empty_initially():
    store = AnnotatedFrameStore()
    assert store.camera_ids() == []


# ---------------------------------------------------------------------------
# AnnotatedFrameStore — happy paths
# ---------------------------------------------------------------------------

def test_put_then_get_returns_same_frame():
    store = AnnotatedFrameStore()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    store.put("cam1", frame)
    result = store.get("cam1")
    assert result is not None
    assert np.array_equal(result, frame)


def test_put_updates_existing_frame():
    store = AnnotatedFrameStore()
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
    store.put("cam1", frame1)
    store.put("cam1", frame2)
    assert np.array_equal(store.get("cam1"), frame2)


def test_camera_ids_returns_all_registered():
    store = AnnotatedFrameStore()
    store.put("cam1", np.zeros((10, 10, 3), dtype=np.uint8))
    store.put("cam2", np.zeros((10, 10, 3), dtype=np.uint8))
    assert sorted(store.camera_ids()) == ["cam1", "cam2"]


def test_multiple_cameras_isolated():
    store = AnnotatedFrameStore()
    f1 = np.zeros((10, 10, 3), dtype=np.uint8)
    f2 = np.ones((10, 10, 3), dtype=np.uint8) * 200
    store.put("cam1", f1)
    store.put("cam2", f2)
    assert np.array_equal(store.get("cam1"), f1)
    assert np.array_equal(store.get("cam2"), f2)


# ---------------------------------------------------------------------------
# mjpeg_stream — unhappy / edge paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mjpeg_stream_empty_store_yields_nothing_quickly():
    """When no frame is available, the generator should yield nothing within a short window."""
    store = AnnotatedFrameStore()
    frames_received = []

    async def collect_with_timeout():
        async for chunk in mjpeg_stream(store, "cam1", max_frames=3, sleep_s=0.0):
            frames_received.append(chunk)

    await asyncio.wait_for(collect_with_timeout(), timeout=1.0)
    assert frames_received == []


# ---------------------------------------------------------------------------
# mjpeg_stream — happy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mjpeg_stream_yields_valid_multipart_frames():
    """Generator yields JPEG multipart chunks for a camera that has a frame."""
    store = AnnotatedFrameStore()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    store.put("cam1", frame)

    chunks = []
    async for chunk in mjpeg_stream(store, "cam1", max_frames=2, sleep_s=0.0):
        chunks.append(chunk)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.startswith(b"--frame\r\n")
        assert b"Content-Type: image/jpeg" in chunk
        assert b"\xff\xd8" in chunk  # JPEG SOI marker


@pytest.mark.asyncio
async def test_mjpeg_stream_chunk_ends_with_crlf():
    store = AnnotatedFrameStore()
    store.put("cam1", np.zeros((32, 32, 3), dtype=np.uint8))
    chunks = []
    async for chunk in mjpeg_stream(store, "cam1", max_frames=1, sleep_s=0.0):
        chunks.append(chunk)
    assert chunks[0].endswith(b"\r\n")
