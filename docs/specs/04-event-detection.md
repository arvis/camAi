# Feature 4: Event Detection

## What it does
Converts raw detections into meaningful tactical events by comparing current state to previous state.

## Steps
1. After each detection, compare new object counts to previous counts for this camera
2. Apply event rules (see table below)
3. Check rate limit: skip if the same event fired for this camera in the last 30 seconds
4. Return list of `Event` objects for any events that fired

## Event rules

| Event | Trigger |
|-------|---------|
| `PERSON_DETECTED` | Person count goes from 0 → 1+ |
| `VEHICLE_ARRIVED` | Car or truck count goes from 0 → 1+ |
| `CROWD_FORMED` | Person count reaches `crowd_threshold` (default: 3) |
| `PERSON_MOVING` | Any visible track has direction ≠ stationary |
| `AREA_CLEARED` | All counts drop to 0 after non-empty state |

## Config (`config.yaml`)
```yaml
detection:
  crowd_threshold: 3
  event_cooldown_s: 30
```

## Event data structure
Each event carries:
- `camera_id`, `camera_name`
- `event_type` (e.g. `CROWD_FORMED`)
- `timestamp`
- `detections` — class counts dict
- `tracks` — list of track IDs + movement directions
- `avg_confidence` — average detection confidence

## Module
`core/events.py` — `EventDetector` class, one instance per camera
