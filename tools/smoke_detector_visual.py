"""tools/smoke_detector_visual.py — Visual webcam smoke-test for Detector + ByteTrack.

Opens a window showing the live annotated feed with bounding boxes, track IDs,
class labels, and movement directions. Press Q to quit.

Usage:
    python tools/smoke_detector_visual.py
    python tools/smoke_detector_visual.py --cam 1 --interval 1
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
    window = "Tactical Vision — press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed.")
            break

        annotated, counts, tracked = detector.process("cam0", frame)

        # HUD: object counts in top-left corner
        y = 20
        for cls, n in sorted(counts.items()):
            cv2.putText(annotated, f"{cls}: {n}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y += 22

        if counts:
            objects = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            directions = " | ".join(
                f"#{obj['track_id']} {obj['class_name']} → {obj['movement_direction']}"
                for obj in tracked
            )
            print(f"[{time.strftime('%H:%M:%S')}] {objects}  |  {directions}")

        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
