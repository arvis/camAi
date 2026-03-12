"""tests/test_routes.py — Unit tests for FastAPI routes.

TDD: unhappy / edge cases first, then happy paths.
Uses FastAPI TestClient (synchronous) and starlette WebSocket test session.
"""

import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.mjpeg import AnnotatedFrameStore
from api.routes import ConnectionManager, create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAMERAS = [{"id": "cam1", "name": "Test-Cam"}]


def make_client(frame_store=None, manager=None):
    fs = frame_store or AnnotatedFrameStore()
    mgr = manager or ConnectionManager()
    app = create_app(fs, mgr, CAMERAS)
    return TestClient(app), fs, mgr


# ---------------------------------------------------------------------------
# GET /stream/{cam_id} — unhappy paths
# ---------------------------------------------------------------------------

def test_stream_unknown_camera_returns_404():
    client, _, _ = make_client()
    resp = client.get("/stream/ghost_cam")
    assert resp.status_code == 404


def test_stream_wrong_method_returns_405():
    client, _, _ = make_client()
    resp = client.post("/stream/cam1")
    assert resp.status_code == 405


# ---------------------------------------------------------------------------
# GET / — happy path
# ---------------------------------------------------------------------------

def test_dashboard_returns_200_html():
    client, _, _ = make_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_dashboard_html_contains_camera_stream():
    """HTML must embed an <img> or <video> pointing at the stream endpoint."""
    client, _, _ = make_client()
    resp = client.get("/")
    assert b"cam1" in resp.content
    assert b"/stream/" in resp.content


# ---------------------------------------------------------------------------
# GET /stream/{cam_id} — happy path (streaming response begins)
# ---------------------------------------------------------------------------

def test_stream_known_camera_returns_200():
    """Stream endpoint returns 200 with correct content-type for a known camera.

    The real mjpeg_stream runs forever, so we patch it with a finite generator
    to avoid blocking the TestClient.
    """
    from unittest.mock import patch

    async def _one_frame(*args, **kwargs):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8\xff\xd9\r\n"

    client, fs, _ = make_client()
    fs.put("cam1", np.zeros((64, 64, 3), dtype=np.uint8))
    with patch("api.routes.mjpeg_stream", _one_frame):
        resp = client.get("/stream/cam1")
    assert resp.status_code == 200
    assert "multipart/x-mixed-replace" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# WebSocket /ws — unhappy path
# ---------------------------------------------------------------------------

def test_ws_disconnect_does_not_crash():
    """Opening a WS and immediately closing should not raise."""
    client, _, _ = make_client()
    with client.websocket_connect("/ws") as ws:
        pass  # just connect and disconnect


# ---------------------------------------------------------------------------
# WebSocket /ws — happy path (broadcast)
# ---------------------------------------------------------------------------

def test_ws_multiple_clients_can_connect():
    """Multiple WebSocket clients can connect simultaneously without error."""
    client, _, _ = make_client()
    with client.websocket_connect("/ws") as ws1:
        with client.websocket_connect("/ws") as ws2:
            pass  # both connected, both disconnect cleanly


# ---------------------------------------------------------------------------
# ConnectionManager — unhappy paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broadcast_with_no_clients_does_not_raise():
    mgr = ConnectionManager()
    await mgr.broadcast({"type": "alert"})  # should not raise


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections():
    """Dead sockets are silently removed, not left accumulating."""
    mgr = ConnectionManager()

    class FakeWS:
        async def send_json(self, data):
            raise RuntimeError("disconnected")

    mgr._connections.append(FakeWS())
    assert len(mgr._connections) == 1
    await mgr.broadcast({"hello": "world"})
    assert len(mgr._connections) == 0
