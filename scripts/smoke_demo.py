"""scripts/smoke_demo.py — End-to-end smoke test / live demo.

Reads camera sources from config.yaml, runs the full detection + event pipeline,
prints a live event feed to the terminal, and saves an annotated video so you can
review exactly what the model saw (bounding boxes, labels, colours by class).

Usage:
    python scripts/smoke_demo.py                              # webcam, saves smoke_output.mp4
    python scripts/smoke_demo.py --source video.mp4           # replay a recording
    python scripts/smoke_demo.py --max-frames 500             # auto-stop
    python scripts/smoke_demo.py --output annotated.mp4       # custom output path
    python scripts/smoke_demo.py --no-output                  # terminal only, no file

Ctrl+C prints a summary and exits cleanly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

# Resolve project root so imports work regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.detector import Detector
from core.events import EventDetector, EventSeverity

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"

_SEVERITY_COLOR = {
    EventSeverity.HIGH_ALERT: _RED + _BOLD,
    EventSeverity.SUSPICIOUS: _YELLOW,
    EventSeverity.NORMAL:     _GREEN,
}

_SEVERITY_LABEL = {
    EventSeverity.HIGH_ALERT: "HIGH_ALERT",
    EventSeverity.SUSPICIOUS: "SUSPICIOUS",
    EventSeverity.NORMAL:     "NORMAL    ",
}

# Overlay colours matching detector.py (BGR)
_OV_HIGH_ALERT = (0, 0, 255)    # red
_OV_SUSPICIOUS = (0, 165, 255)  # orange
_OV_NORMAL     = (0, 200, 0)    # green


def _fmt_event(ev) -> str:
    color = _SEVERITY_COLOR[ev.severity]
    label = _SEVERITY_LABEL[ev.severity]
    ts = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
    counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(ev.detections.items()))
    return (
        f"{_DIM}{ts}{_RESET}  "
        f"{color}[{label}]{_RESET}  "
        f"{_BOLD}{ev.event_type:<22}{_RESET}  "
        f"cam={ev.camera_name}  "
        f"conf={ev.avg_confidence:.2f}  "
        f"[{counts_str}]"
    )


def _overlay_events(frame, active_events: list) -> None:
    """Burn the last few active events onto the frame as a HUD strip."""
    y = 30
    for ev in active_events[-5:]:  # show latest 5 on-screen
        if ev.severity == EventSeverity.HIGH_ALERT:
            bg, fg = _OV_HIGH_ALERT, (255, 255, 255)
        elif ev.severity == EventSeverity.SUSPICIOUS:
            bg, fg = _OV_SUSPICIOUS, (0, 0, 0)
        else:
            bg, fg = _OV_NORMAL, (0, 0, 0)
        text = f"{ev.event_type}  conf={ev.avg_confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (8, y - th - 4), (8 + tw + 6, y + 4), bg, -1)
        cv2.putText(frame, text, (11, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)
        y += th + 10


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tactical Vision AI — smoke demo")
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    parser.add_argument("--source", default=None,
                        help="Override camera source (path or integer index)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = run until Ctrl+C or end of file)")
    parser.add_argument("--output", default="smoke_output.mp4",
                        help="Output annotated video path (default: smoke_output.mp4)")
    parser.add_argument("--no-output", action="store_true",
                        help="Disable video output — terminal only")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg.get("detection", {})

    detector = Detector(
        interval_frames=det_cfg.get("interval_frames", 3),
        weapon_model_path=det_cfg.get("weapon_model_path") or None,
    )

    cam_cfg  = cfg["cameras"][0]
    cam_id   = cam_cfg["id"]
    cam_name = cam_cfg["name"]
    source   = args.source or cam_cfg["url"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    event_det = EventDetector(
        camera_id=cam_id,
        camera_name=cam_name,
        crowd_threshold       = det_cfg.get("crowd_threshold", 3),
        event_cooldown_s      = det_cfg.get("event_cooldown_s", 30.0),
        high_alert_cooldown_s = det_cfg.get("high_alert_cooldown_s", 10.0),
        loitering_timeout_s   = det_cfg.get("loitering_timeout_s", 45.0),
        running_threshold_px  = det_cfg.get("running_threshold_px", 80.0),
        abandoned_timeout_s   = det_cfg.get("abandoned_timeout_s", 60.0),
        abandoned_proximity_px= det_cfg.get("abandoned_proximity_px", 150.0),
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {source}", file=sys.stderr)
        sys.exit(1)

    # Probe dimensions for VideoWriter
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if not args.no_output:
        out_path = str(Path(args.output).resolve())
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, src_fps, (w, h))
        if not writer.isOpened():
            print(f"WARNING: could not open VideoWriter for {out_path} — disabling output",
                  file=sys.stderr)
            writer = None

    print(f"\n{_BOLD}{_CYAN}Tactical Vision AI — smoke demo{_RESET}")
    print(f"Camera : {cam_name} ({source})  {w}x{h} @ {src_fps:.0f} fps")
    print(f"Model  : yolov8n.pt  |  interval: every {det_cfg.get('interval_frames', 3)} frames")
    print(f"Crowd  : {det_cfg.get('crowd_threshold', 3)} persons  "
          f"|  cooldown NORMAL={det_cfg.get('event_cooldown_s', 30)}s  "
          f"HIGH={det_cfg.get('high_alert_cooldown_s', 10)}s")
    if writer:
        print(f"Output : {out_path}")
    print(f"{_DIM}Press Ctrl+C to stop{_RESET}\n")
    print(f"{'Time':8}  {'Severity':28}  {'Event':<22}  Details")
    print("-" * 90)

    frame_count  = 0
    event_count  = 0
    event_log: list = []
    recent_events: list = []   # shown as HUD on video
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"{_DIM}End of stream.{_RESET}")
                break

            frame_count += 1
            annotated, counts, tracked = detector.process(cam_id, frame)
            events = event_det.process(counts, tracked)

            for ev in events:
                event_count += 1
                event_log.append(ev)
                recent_events.append(ev)
                print(_fmt_event(ev))

            # Burn HUD onto annotated frame and write
            if writer is not None:
                _overlay_events(annotated, recent_events)
                writer.write(annotated)

            # Periodic heartbeat
            if frame_count % 100 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"{_DIM}  -- {frame_count} frames  {fps:.1f} fps  "
                    f"{event_count} events so far --{_RESET}"
                )

            if args.max_frames and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\nAnnotated video saved → {out_path}")

    elapsed = time.time() - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n{_BOLD}--- Summary ---{_RESET}")
    print(f"Frames : {frame_count}  ({fps:.1f} fps avg)")
    print(f"Events : {event_count}")
    if event_log:
        from collections import Counter
        counts_by_type = Counter(e.event_type for e in event_log)
        for etype, n in counts_by_type.most_common():
            ev = next(e for e in event_log if e.event_type == etype)
            color = _SEVERITY_COLOR[ev.severity]
            print(f"  {color}{etype:<25}{_RESET}  x{n}")
    print()


if __name__ == "__main__":
    main()
