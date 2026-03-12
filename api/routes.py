"""api/routes.py — FastAPI application factory with dashboard routes.

Endpoints
---------
GET /                      Dashboard HTML
GET /stream/{cam_id}       Annotated MJPEG stream (multipart/x-mixed-replace)
WS  /ws                    Real-time alert/report push to browser
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from api.mjpeg import AnnotatedFrameStore, mjpeg_stream

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts JSON messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send message to all connected clients; silently remove dead sockets."""
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def create_app(
    frame_store: AnnotatedFrameStore,
    manager: ConnectionManager,
    cameras: list[dict],
) -> FastAPI:
    """Create and return the FastAPI application.

    Parameters
    ----------
    frame_store : AnnotatedFrameStore
        Shared store populated by camera processing threads.
    manager : ConnectionManager
        Shared WebSocket connection manager.
    cameras : list[dict]
        Camera config list (each dict has 'id' and 'name').
    """
    app = FastAPI(title="Tactical Vision AI")
    _cam_ids = {c["id"] for c in cameras}

    _cam_json = json.dumps([{"id": c["id"], "name": c["name"]} for c in cameras])

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        html = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace("CAM_DATA_PLACEHOLDER", _cam_json)
        return HTMLResponse(content=html)

    @app.get("/stream/{cam_id}")
    async def stream(cam_id: str):
        if cam_id not in _cam_ids:
            raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found")
        return StreamingResponse(
            mjpeg_stream(frame_store, cam_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                # Keep connection alive; client may send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(ws)
        except Exception:
            manager.disconnect(ws)

    return app
