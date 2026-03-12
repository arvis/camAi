"""main.py — Tactical Vision AI server entry point.

Starts camera processing threads, wires the detection pipeline, and serves
the FastAPI dashboard over uvicorn.

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import yaml
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from api.mjpeg import AnnotatedFrameStore
from api.routes import ConnectionManager, create_app
from core.alerts import AlertSender
from core.detector import Detector
from core.events import AlertLevel, EventDetector
from core.intelligence import SituationReporter
from core.stream import CameraStream
from llm.factory import get_provider, get_vision_provider

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Camera processing thread
# ---------------------------------------------------------------------------

def _camera_loop(
    cam_cfg: dict,
    detector: Detector,
    event_detector: EventDetector,
    alert_sender: AlertSender,
    reporter: SituationReporter,
    frame_store: AnnotatedFrameStore,
    ws_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """Background thread: read frames → detect → store annotated → fire events."""
    cam_id = cam_cfg["id"]
    cam_name = cam_cfg["name"]
    stream = CameraStream(cam_id, cam_name, cam_cfg["url"])
    stream.start()
    logger.info("Camera thread started: %s (%s)", cam_name, cam_cfg["url"])

    while not stop_event.is_set():
        frame = stream.buffer.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        annotated, counts, tracked = detector.process(cam_id, frame)
        frame_store.put(cam_id, annotated)

        events = event_detector.process(counts, tracked)
        for event in events:
            # Attach JPEG snapshot for PERSON_DETECTED vision analysis
            if event.event_type == "PERSON_DETECTED":
                ok, jpeg = cv2.imencode(".jpg", frame)
                if ok:
                    event.frame_snapshot = jpeg.tobytes()

            reporter.record_event(event)

            # Slack alert (async, fire-and-forget from thread)
            asyncio.run_coroutine_threadsafe(alert_sender.send_alert(event), loop)

            # Push to WebSocket clients
            msg = {
                "type": "alert",
                "camera_name": event.camera_name,
                "event_type": event.event_type,
                "objects": event.detections,
                "severity": event.severity.value,
                "timestamp": datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S"),
            }
            loop.call_soon_threadsafe(ws_queue.put_nowait, msg)

    stream.stop()
    logger.info("Camera thread stopped: %s", cam_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_app(config_path: str = "config.yaml") -> FastAPI:
    """Build and return the FastAPI app. Called once at startup."""
    config = _load_config(config_path)
    cameras = config["cameras"]
    det_cfg = config["detection"]
    rep_cfg = config["report"]

    frame_store = AnnotatedFrameStore()
    manager = ConnectionManager()
    alert_sender = AlertSender.from_env()

    detector = Detector(
        interval_frames=det_cfg["interval_frames"],
        weapon_model_path=det_cfg.get("weapon_model_path"),
    )

    event_detectors = {
        c["id"]: EventDetector(
            camera_id=c["id"],
            camera_name=c["name"],
            crowd_threshold=det_cfg["crowd_threshold"],
            event_cooldown_s=det_cfg["event_cooldown_s"],
            high_alert_cooldown_s=det_cfg["high_alert_cooldown_s"],
            loitering_timeout_s=det_cfg["loitering_timeout_s"],
            running_threshold_px=det_cfg["running_threshold_px"],
            abandoned_timeout_s=det_cfg["abandoned_timeout_s"],
            abandoned_proximity_px=det_cfg["abandoned_proximity_px"],
        )
        for c in cameras
    }

    llm = get_provider()
    vision = get_vision_provider()

    # Placeholder — reporter is created inside lifespan so we can wire the callback
    reporter_holder: list[SituationReporter] = []
    ws_queue_holder: list[asyncio.Queue] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        ws_queue: asyncio.Queue = asyncio.Queue()
        ws_queue_holder.append(ws_queue)

        async def report_cb(text: str) -> None:
            ts = datetime.now().strftime("%H:%M:%S")
            ws_queue.put_nowait({"type": "report", "text": text, "timestamp": ts})

        reporter = SituationReporter(
            llm=llm,
            alert_sender=alert_sender,
            interval_s=rep_cfg["interval_seconds"],
            camera_names={c["id"]: c["name"] for c in cameras},
            vision_provider=vision,
            alert_level=AlertLevel[rep_cfg["alert_level"].upper()],
            report_callback=report_cb,
        )
        reporter_holder.append(reporter)

        stop_event = threading.Event()
        threads: list[threading.Thread] = []
        for cam_cfg in cameras:
            t = threading.Thread(
                target=_camera_loop,
                args=(
                    cam_cfg,
                    detector,
                    event_detectors[cam_cfg["id"]],
                    alert_sender,
                    reporter,
                    frame_store,
                    ws_queue,
                    loop,
                    stop_event,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        reporter_task = asyncio.create_task(reporter.run())

        async def _ws_broadcaster() -> None:
            while True:
                msg = await ws_queue.get()
                await manager.broadcast(msg)

        broadcaster_task = asyncio.create_task(_ws_broadcaster())

        logger.info("Tactical Vision AI ready — http://localhost:8000")

        try:
            yield
        finally:
            stop_event.set()
            reporter_task.cancel()
            broadcaster_task.cancel()

    app = create_app(frame_store, manager, cameras)
    app.router.lifespan_context = lifespan
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Tactical Vision AI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    app = build_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
