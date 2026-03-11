# Feature 2: Object Detection

## What it does
Runs YOLOv8 on each frame to find people and vehicles. Draws visual overlays on the frame.

## Steps
1. Load `yolov8n.pt` model once at startup (auto-downloads on first run)
2. For every 3rd frame from each camera, run detection
3. Filter results to these classes: person, car, truck, motorcycle, bicycle
4. Draw bounding boxes + class label + confidence score on the frame using OpenCV
5. Return: annotated frame + count per class (e.g. `{"person": 2, "car": 1}`)

## Config (`config.yaml`)
```yaml
detection:
  interval_frames: 3    # run detection every N frames
```

## Module
`core/detector.py` — `Detector` class wrapping `ultralytics.YOLO`

## Notes
- Model: `yolov8n.pt` (nano) — fastest, works on CPU
- Upgrade to `yolov8s.pt` if GPU is available
