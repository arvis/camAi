"""tools/smoke_detector.py — Quick webcam smoke-test for Detector + ByteTrack.

Reads from webcam (index 0), runs the detector every N frames, and prints
tracked objects to stdout. Press Ctrl+C to stop.

Usage:
    python tools/smoke_detector.py
    python tools/smoke_detector.py --cam 1 --interval 1
"""

import argparse
import sys
import time

import cv2

sys.path.insert(0, ".")
from core.detector import Detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--interval", type=int, default=3,
                        help="Detect every N frames (default 3)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam {args.cam}")
        sys.exit(1)

    detector = Detector(interval_frames=args.interval)
    print(f"Running on webcam {args.cam}, detection every {args.interval} frames. Ctrl+C to stop.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed — is the webcam connected?")
                break

            _, counts, tracked = detector.process("cam0", frame)

            if counts:
                objects = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                directions = " | ".join(
                    f"#{obj['track_id']} {obj['class_name']} → {obj['movement_direction']}"
                    for obj in tracked
                )
                print(f"[{time.strftime('%H:%M:%S')}] {objects}  |  {directions}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
