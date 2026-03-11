"""Smoke test: grab one frame from webcam, run detector, print counts.

Run from project root:
    .venv/Scripts/python smoke_tests/detect.py
"""

import cv2
from core.detector import Detector

CAM_INDEX = 0  # change to a video path if needed, e.g. "test_footage/video.mp4"

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {CAM_INDEX!r}")
        return

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("ERROR: failed to read frame")
        return

    print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")

    det = Detector(interval_frames=1)
    annotated, counts = det.process("cam1", frame)

    print("Detections:", counts if counts else "(none)")

    try:
        title = "smoke_detect — press SPACE to close"
        cv2.imshow(title, annotated)
        while cv2.waitKey(100) != ord(" "):
            pass
        cv2.destroyAllWindows()
    except Exception:
        out = "smoke_tests/detect_output.jpg"
        cv2.imwrite(out, annotated)
        print(f"No display available — saved annotated frame to {out}")


if __name__ == "__main__":
    main()
