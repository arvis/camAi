# Feature 1: Phone Stream Ingestion

## What it does
Reads live MJPEG video streams from phones running IP Webcam over the shared WiFi network (e.g. Starlink router in the field).
Each phone = one camera source (drone, body cam, checkpoint, etc.).

## Phone setup
1. Install **IP Webcam** (Android) or **EpocCam** (iOS) on each phone
2. Connect phone to the shared WiFi network
3. Start the stream — note the IP shown (e.g. `http://192.168.1.10:8080/video`)
4. Add the URL to `config.yaml`

## Steps
1. Load camera list from `config.yaml`
2. For each camera, start a `threading.Thread`
3. Thread opens stream: `cv2.VideoCapture(url)`
4. Thread reads frames in a loop, stores the latest frame in a shared `CameraBuffer`
5. Detection pipeline polls the buffer
6. On disconnect: retry with exponential backoff (5s → 10s → 20s → max 60s)

## Config (`config.yaml`)
```yaml
cameras:
  - id: cam1
    name: Drone-1
    url: http://192.168.1.10:8080/video
  - id: cam2
    name: Checkpoint
    url: http://192.168.1.11:8080/video
  - id: cam3
    name: Drone-2
    url: http://192.168.1.12:8080/video

detection:
  interval_frames: 3
  crowd_threshold: 3
  event_cooldown_s: 30
```

## Fallback for testing (no phones)
```yaml
cameras:
  - id: cam1
    name: Test-Cam
    url: 0                          # local webcam
  - id: cam2
    name: Test-Video
    url: ./test_footage/drone.mp4   # video file
```

## Module
`core/stream.py` — `CameraStream(camera_id, name, url)` class, one instance per camera
