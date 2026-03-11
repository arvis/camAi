# Feature 3: Object Tracking

## What it does
Gives each detected object a persistent ID across frames and estimates its movement direction.

## Steps
1. Use `model.track(frame, tracker="bytetrack.yaml", persist=True)` — built into Ultralytics, no extra install
2. Each object gets a stable `track_id` integer that persists across frames
3. Store the last 10 centroid positions per track ID
4. Compute movement direction from the centroid delta: N / S / E / W / stationary
5. Pass track IDs and directions into event detection and alert messages

## Movement direction rules
- Compute delta between oldest and newest stored centroid
- If total movement < 15px: `stationary`
- Otherwise map to nearest compass direction (N/S/E/W)

## Output per tracked object
- `track_id: int`
- `movement_direction: str` — one of: N, S, E, W, stationary, unknown

## Module
Part of `core/detector.py` — tracking runs inside `Detector.process_frame()`
