"""tools/smoke_person_alert.py — Live webcam: person detection + Slack alerts + 2-min reports.

Flow (alert_level=high, default):
  1. Open webcam, run YOLOv8 + ByteTrack on each frame.
  2. On PERSON_DETECTED: send NORMAL alert to Slack (rate-limited by AlertSender).
  3. Vision-check the frame for military/armed indicators via LLM.
     → CRITICAL MILITARY_PERSON_DETECTED alert if detected.
  4. Every 2 minutes: AI situation report posted to Slack.

At alert_level=normal steps 3 is skipped entirely.

Usage:
    python tools/smoke_person_alert.py
    python tools/smoke_person_alert.py --cam 1 --alert-level normal
    python tools/smoke_person_alert.py --report-interval 30

Requires:
    .env with SLACK_WEBHOOK_URL and (OPENAI_API_KEY or GEMINI_API_KEY)
    LLM_PROVIDER=openai or LLM_PROVIDER=gemini  (Anthropic/Ollama have no vision support)
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
from core.events import AlertLevel, Event, EventSeverity
from core.intelligence import SituationReporter
from llm.factory import get_provider, get_vision_provider

_CAM_ID = "cam0"
_CAM_NAME = "Webcam"

_VISION_PROMPT = (
    "You are a tactical analyst reviewing a surveillance frame. "
    "Answer YES if ANY of the following are visible on the person: "
    "helmet, tactical vest, military uniform, combat boots, weapon, firearm, "
    "ammunition pouches, or any other military/combat equipment. "
    "A single indicator is enough to answer YES. "
    "Answer NO only if none of these indicators are present. "
    "Start your response with YES or NO, then briefly state which indicator(s) you saw. "
    "Also note any colored IFF identification bands (red/blue armbands, headbands)."
)

_MILITARY_KEYWORDS = frozenset({
    "military", "armed", "tactical", "rifle", "pistol",
    "gun", "firearm", "soldier", "combatant", "armband", "iff",
})


def _is_military(analysis: str) -> bool:
    """Return True only when the vision LLM affirmatively identifies military/armed indicators."""
    stripped = analysis.strip().lower()
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    # Fallback for unstructured responses
    return any(kw in stripped for kw in _MILITARY_KEYWORDS)


def _encode_jpeg(frame) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live webcam: person alerts + military vision check + 2-min situation reports."
    )
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--interval", type=int, default=3,
                        help="Detect every N frames (default 3)")
    parser.add_argument("--alert-level", choices=["high", "normal"], default="high",
                        help="Alert level: 'high' enables military vision check (default: high)")
    parser.add_argument("--report-interval", type=float, default=120.0,
                        help="Seconds between situation reports (default 120)")
    args = parser.parse_args()

    alert_level = AlertLevel.HIGH if args.alert_level == "high" else AlertLevel.NORMAL

    # ── LLM providers ──────────────────────────────────────────────────────────
    llm = get_provider()
    vision = get_vision_provider() if alert_level == AlertLevel.HIGH else None
    if alert_level == AlertLevel.HIGH and vision is None:
        print(
            "WARNING: no vision provider available (set LLM_PROVIDER=openai or gemini).\n"
            "         Military vision checks will be skipped."
        )

    # ── alert sender ───────────────────────────────────────────────────────────
    sender = AlertSender.from_env()
    if not sender.webhook_url:
        print("WARNING: SLACK_WEBHOOK_URL not set in .env — Slack posts will be skipped.")

    # ── situation reporter ─────────────────────────────────────────────────────
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        interval_s=args.report_interval,
        camera_names={_CAM_ID: _CAM_NAME},
        vision_provider=None,  # vision checks done inline below; reporter only posts summary
        alert_level=alert_level,
    )

    # ── webcam ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam {args.cam}")
        sys.exit(1)

    detector = Detector(interval_frames=args.interval)
    window = "Tactical Vision (person alert) — press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    report_task = asyncio.create_task(reporter.run())
    loop = asyncio.get_event_loop()

    print(f"Alert level : {alert_level.value.upper()}")
    print(f"Report every: {args.report_interval}s")
    print("Press Q to quit, C to clear emergency.\n")

    last_person_count = 0
    emergency = False

    try:
        while True:
            # Run blocking cap.read in executor so the event loop stays responsive
            ok, frame = await loop.run_in_executor(None, cap.read)
            if not ok or frame is None:
                print("Frame read failed — stopping.")
                break

            annotated, counts, tracked = detector.process(_CAM_ID, frame)
            person_count = counts.get("person", 0)

            # HUD: object counts in top-left corner
            y = 20
            for cls, n in sorted(counts.items()):
                cv2.putText(annotated, f"{cls}: {n}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                y += 22

            if emergency:
                cv2.putText(annotated, "!! EMERGENCY — press C to clear !!", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # ── person detected → alert (skip all when emergency active) ──────
            elif person_count > 0:
                snapshot = _encode_jpeg(frame)
                avg_conf = (
                    sum(t.get("confidence", 0.0) for t in tracked) / len(tracked)
                    if tracked else 0.0
                )

                event = Event(
                    camera_id=_CAM_ID,
                    camera_name=_CAM_NAME,
                    event_type="PERSON_DETECTED",
                    severity=EventSeverity.NORMAL,
                    timestamp=time.time(),
                    detections=dict(counts),
                    tracks=list(tracked),
                    avg_confidence=avg_conf,
                    frame_snapshot=snapshot,
                )

                await sender.send_alert(event)
                reporter.record_event(event)

                # ── military vision check: only when count increases ───────────
                if vision is not None and snapshot and person_count > last_person_count:
                    try:
                        analysis = await vision.analyze_image(snapshot, _VISION_PROMPT)
                        print(f"  Vision: {analysis}")
                        if _is_military(analysis):
                            mil_event = Event(
                                camera_id=_CAM_ID,
                                camera_name=_CAM_NAME,
                                event_type="MILITARY_PERSON_DETECTED",
                                severity=EventSeverity.CRITICAL,
                                timestamp=time.time(),
                                detections=dict(counts),
                                tracks=list(tracked),
                                avg_confidence=avg_conf,
                                frame_snapshot=snapshot,
                            )
                            await sender.send_alert(mil_event)
                            reporter.record_event(mil_event)
                            emergency = True
                            print(f"  --> CRITICAL: EMERGENCY DECLARED — press C to clear")
                    except Exception as exc:
                        print(f"  Vision analysis failed: {exc}")

            last_person_count = person_count

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and emergency:
                emergency = False
                print(f"[{time.strftime('%H:%M:%S')}] Emergency cleared.")

    finally:
        report_task.cancel()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
