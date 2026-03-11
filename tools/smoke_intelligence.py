"""tools/smoke_intelligence.py — End-to-end smoke test for Feature 06.

Captures N seconds of webcam footage, runs YOLOv8 + EventDetector on each
frame, feeds real events into SituationReporter, then calls the LLM and posts
the generated report to Slack.

Usage:
    python tools/smoke_intelligence.py
    python tools/smoke_intelligence.py --cam 1 --duration 15 --interval 3
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import cv2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

from core.alerts import AlertSender
from core.detector import Detector
from core.events import EventDetector
from core.intelligence import SituationReporter
from llm.factory import get_provider, get_vision_provider

CAM_ID = "cam1"
CAM_NAME = "Webcam"


def capture_events(cam_index: int, duration_s: int, interval_frames: int) -> SituationReporter:
    """Capture webcam frames, run detection + event detection, return a populated reporter."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam {cam_index}")
        sys.exit(1)

    detector = Detector(interval_frames=interval_frames)
    event_detector = EventDetector(camera_id=CAM_ID, camera_name=CAM_NAME)

    llm = get_provider()
    vision = get_vision_provider()
    sender = AlertSender.from_env()
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        camera_names={CAM_ID: CAM_NAME},
        vision_provider=vision,
    )
    if vision:
        print("Vision analysis enabled — military/IFF check active.")
    else:
        print("No vision provider — set LLM_PROVIDER=openai or gemini for military check.")

    print(f"Recording {CAM_NAME} for {duration_s}s (detection every {interval_frames} frames)...")
    deadline = time.monotonic() + duration_s
    total_frames = 0
    total_events = 0

    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed — is the webcam connected?")
                break

            total_frames += 1
            _, counts, tracked = detector.process(CAM_ID, frame)
            events = event_detector.process(counts, tracked)

            for ev in events:
                if ev.event_type == "PERSON_DETECTED" and vision:
                    ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok_enc:
                        ev.frame_snapshot = buf.tobytes()
                reporter.record_event(ev)
                total_events += 1
                print(f"  [{time.strftime('%H:%M:%S')}] {ev.event_type} (severity={ev.severity.value})")
    finally:
        cap.release()

    print(f"\nCaptured {total_frames} frames, fired {total_events} events.")
    return reporter


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--duration", type=int, default=10, help="Seconds to record (default 10)")
    parser.add_argument("--interval", type=int, default=3, help="Detect every N frames (default 3)")
    args = parser.parse_args()

    reporter = capture_events(args.cam, args.duration, args.interval)

    print("Calling LLM and posting to Slack...")
    await reporter._generate_and_post()
    print("Done. Check Slack for the situation report.")


if __name__ == "__main__":
    asyncio.run(main())
