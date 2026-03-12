# Feature 4: Event Detection

## What it does
Converts raw detections into meaningful tactical events by comparing current state to
previous state. Events have three severity levels — NORMAL, SUSPICIOUS, HIGH_ALERT —
which drive alert formatting and cooldown rules.

## Severity levels

| Severity | Description | Cooldown |
|----------|-------------|----------|
| `NORMAL` | Routine observation | `event_cooldown_s` (default 30 s) |
| `SUSPICIOUS` | Behaviour worth watching | `event_cooldown_s` (default 30 s) |
| `HIGH_ALERT` | Immediate threat indicator | `high_alert_cooldown_s` (default 10 s) |

HIGH_ALERT events fire far more frequently so operators are not silently deprived of
repeated threat signals during an incident.

## Event rules

| Event | Trigger | Severity |
|-------|---------|----------|
| `PERSON_DETECTED` | Person count 0 → 1+ | NORMAL |
| `VEHICLE_ARRIVED` | Car or truck count 0 → 1+ | NORMAL |
| ~~`CROWD_FORMED`~~ | ~~Person count ≥ `crowd_threshold`~~ | ~~NORMAL~~ — **disabled**: fires too aggressively; threshold tuning needed before re-enabling |
| `PERSON_MOVING` | Any tracked person direction ≠ stationary | NORMAL |
| `AREA_CLEARED` | All counts drop to 0 after non-empty state | NORMAL |
| `LOITERING` | Same person track_id stationary for > `loitering_timeout_s` | SUSPICIOUS |
| `RUNNING` | Person track `velocity_px` > `running_threshold_px` per detection | SUSPICIOUS |
| `ABANDONED_OBJECT` | Bag track visible > `abandoned_timeout_s` with no person within `abandoned_proximity_px` px | SUSPICIOUS |
| `KNIFE_DETECTED` | `knife` class detected by base YOLO model | HIGH_ALERT |
| `PISTOL_DETECTED` | `pistol` class detected by weapon model | HIGH_ALERT |
| `RIFLE_DETECTED` | `rifle` or `assault_rifle` class detected by weapon model | HIGH_ALERT |

### Notes on specific rules

**Abandoned object** — bag-type classes (`backpack`, `handbag`, `suitcase`) are tracked
by `track_id`. The timer starts when the track first appears. It resets if a person
centroid comes within `abandoned_proximity_px` pixels. Fires once the timeout elapses.

**Loitering** — timer resets when the track's `movement_direction` changes away from
`stationary`. Does not fire again until the track moves and becomes stationary again.

**Weapon events (knife, pistol, rifle)** — `knife` is a standard COCO class; no extra
model needed. Pistol/rifle require `weapon_model_path` (a YOLOv8-format `.pt` file).
Free pre-trained models with `pistol`, `rifle`, `assault_rifle` classes are available
on Roboflow Universe — download any yolov8-format export and point `weapon_model_path`
at it. Leave `weapon_model_path: null` to skip firearm detection silently.

## Config (`config.yaml`)

```yaml
detection:
  # Crowd threshold — change this freely to tune sensitivity
  crowd_threshold: 3          # persons needed to trigger CROWD_FORMED

  # Cooldowns
  event_cooldown_s: 30        # NORMAL and SUSPICIOUS events
  high_alert_cooldown_s: 10   # HIGH_ALERT events (weapons etc.)

  # Suspicious behaviour timers
  loitering_timeout_s: 45     # seconds stationary before LOITERING fires
  running_threshold_px: 80    # centroid pixels/detection interval = running
  abandoned_timeout_s: 60     # seconds bag visible alone before ABANDONED_OBJECT
  abandoned_proximity_px: 150 # px radius — person within this = bag not abandoned

  # Weapon model (optional)
  weapon_model_path: null     # path to .pt with pistol/rifle/assault_rifle classes
```

## Detector changes required (`core/detector.py`)

These changes are needed before `events.py` can implement the new rules:

1. **Expand `TARGET_CLASSES`** — add `backpack`, `handbag`, `suitcase`, `knife`
2. **Add `velocity_px` to tracked objects** — Euclidean distance between current
   centroid and the previous centroid (0.0 if first appearance). Used by running and
   abandoned-object logic.
3. **Optional weapon model** — `Detector` accepts `weapon_model_path: str | None`.
   If set, loads a second YOLO instance. On each detection frame, both models run and
   their results are merged into the same `counts` dict and `tracked` list.
4. **Overlay colour by severity class** —
   - weapons → red box `(0, 0, 255)`
   - bag types → yellow box `(0, 215, 255)`
   - everything else → green (unchanged)

## Event data structure

Each `Event` carries:
- `camera_id`, `camera_name`
- `event_type` (e.g. `CROWD_FORMED`)
- `severity` (`EventSeverity.NORMAL | SUSPICIOUS | HIGH_ALERT`)
- `timestamp`
- `detections` — class counts dict
- `tracks` — list of track IDs + movement directions + velocity_px
- `avg_confidence` — average detection confidence

## `EventDetector` state per camera

`EventDetector` (one instance per camera) must maintain:

```python
prev_counts: dict[str, int]           # last detection's class counts
track_stationary_since: dict[int, float]   # track_id → timestamp when went stationary
bag_first_seen: dict[int, float]           # track_id → timestamp first detected
bag_last_person_nearby: dict[int, float]   # track_id → timestamp person last nearby
```

## Module
`core/events.py` — `EventDetector` class, one instance per camera
