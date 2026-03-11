"""core/events.py — Rules-based event detection state machine.

EventDetector (one instance per camera) compares current detection state
to previous state and emits Event objects for meaningful tactical changes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EventSeverity(Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_ALERT = "HIGH_ALERT"


@dataclass
class Event:
    camera_id: str
    camera_name: str
    event_type: str
    severity: EventSeverity
    timestamp: float
    detections: dict[str, int]
    tracks: list[dict]
    avg_confidence: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BAG_CLASSES = {"backpack", "handbag", "suitcase"}
_VEHICLE_CLASSES = {"car", "truck"}


# ---------------------------------------------------------------------------
# EventDetector
# ---------------------------------------------------------------------------

class EventDetector:
    """State machine that converts raw detections into tactical events.

    Parameters
    ----------
    camera_id, camera_name : str
        Identity of the camera this instance monitors.
    crowd_threshold : int
        Person count that triggers CROWD_FORMED.
    event_cooldown_s : float
        Minimum seconds between NORMAL/SUSPICIOUS events of the same type.
    high_alert_cooldown_s : float
        Minimum seconds between HIGH_ALERT events of the same type.
    loitering_timeout_s : float
        Seconds a person track must be stationary before LOITERING fires.
    running_threshold_px : float
        velocity_px must exceed this to trigger RUNNING.
    abandoned_timeout_s : float
        Seconds a bag must be alone before ABANDONED_OBJECT fires.
    abandoned_proximity_px : float
        Pixel radius within which a person resets the abandoned timer.
    """

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        crowd_threshold: int = 3,
        event_cooldown_s: float = 30.0,
        high_alert_cooldown_s: float = 10.0,
        loitering_timeout_s: float = 45.0,
        running_threshold_px: float = 80.0,
        abandoned_timeout_s: float = 60.0,
        abandoned_proximity_px: float = 150.0,
    ) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.crowd_threshold = crowd_threshold
        self.event_cooldown_s = event_cooldown_s
        self.high_alert_cooldown_s = high_alert_cooldown_s
        self.loitering_timeout_s = loitering_timeout_s
        self.running_threshold_px = running_threshold_px
        self.abandoned_timeout_s = abandoned_timeout_s
        self.abandoned_proximity_px = abandoned_proximity_px

        # Persistent state (per spec)
        self.prev_counts: dict[str, int] = {}
        self.track_stationary_since: dict[int, float] = {}
        self.bag_first_seen: dict[int, float] = {}
        self.bag_last_person_nearby: dict[int, float] = {}

        # Cooldown tracking: event_type → last fired timestamp
        self._last_event_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        counts: dict[str, int],
        tracked: list[dict],
        now: float | None = None,
    ) -> list[Event]:
        """Evaluate counts + tracked objects against event rules.

        Parameters
        ----------
        counts : dict[str, int]
            Per-class detection counts from the detector.
        tracked : list[dict]
            Per-object tracking data (must include velocity_px field).
        now : float | None
            Current timestamp (uses time.time() if None).

        Returns
        -------
        list[Event]
            Events that fired this frame (may be empty).
        """
        if now is None:
            now = time.time()

        events: list[Event] = []
        avg_conf = _avg_confidence(tracked)

        def emit(event_type: str, severity: EventSeverity) -> None:
            ev = self._try_emit(event_type, severity, counts, tracked, avg_conf, now)
            if ev is not None:
                events.append(ev)

        # ---- Count-based transitions ----
        prev_person = self.prev_counts.get("person", 0)
        curr_person = counts.get("person", 0)

        prev_vehicle = sum(self.prev_counts.get(c, 0) for c in _VEHICLE_CLASSES)
        curr_vehicle = sum(counts.get(c, 0) for c in _VEHICLE_CLASSES)

        if prev_person == 0 and curr_person >= 1:
            emit("PERSON_DETECTED", EventSeverity.NORMAL)

        if prev_vehicle == 0 and curr_vehicle >= 1:
            emit("VEHICLE_ARRIVED", EventSeverity.NORMAL)

        if curr_person >= self.crowd_threshold:
            emit("CROWD_FORMED", EventSeverity.NORMAL)

        # AREA_CLEARED: all counts drop to 0 after non-empty previous state
        if counts == {} and self.prev_counts:
            emit("AREA_CLEARED", EventSeverity.NORMAL)

        # ---- Per-track rules ----
        person_centroids: list[tuple[float, float]] = []
        for obj in tracked:
            cls = obj.get("class_name", "")
            tid = obj.get("track_id")
            direction = obj.get("movement_direction", "unknown")
            velocity = obj.get("velocity_px", 0.0)
            bbox = obj.get("bbox", (0, 0, 0, 0))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            if cls == "person":
                person_centroids.append((cx, cy))

                # PERSON_MOVING
                if direction not in ("stationary", "unknown"):
                    emit("PERSON_MOVING", EventSeverity.NORMAL)

                # RUNNING
                if velocity > self.running_threshold_px:
                    emit("RUNNING", EventSeverity.SUSPICIOUS)

                # LOITERING — update stationary timer
                if tid is not None:
                    if direction == "stationary":
                        if tid not in self.track_stationary_since:
                            self.track_stationary_since[tid] = now
                        elif now - self.track_stationary_since[tid] > self.loitering_timeout_s:
                            emit("LOITERING", EventSeverity.SUSPICIOUS)
                    else:
                        self.track_stationary_since.pop(tid, None)

            elif cls in _BAG_CLASSES:
                if tid is not None:
                    if tid not in self.bag_first_seen:
                        self.bag_first_seen[tid] = now

        # Update bag proximity with person centroids gathered above
        for obj in tracked:
            cls = obj.get("class_name", "")
            tid = obj.get("track_id")
            if cls not in _BAG_CLASSES or tid is None:
                continue
            bbox = obj.get("bbox", (0, 0, 0, 0))
            bcx = (bbox[0] + bbox[2]) / 2.0
            bcy = (bbox[1] + bbox[3]) / 2.0
            for pcx, pcy in person_centroids:
                if math.hypot(pcx - bcx, pcy - bcy) < self.abandoned_proximity_px:
                    self.bag_last_person_nearby[tid] = now
                    break

        # ABANDONED_OBJECT check
        for obj in tracked:
            cls = obj.get("class_name", "")
            tid = obj.get("track_id")
            if cls not in _BAG_CLASSES or tid is None:
                continue
            if tid not in self.bag_first_seen:
                continue
            effective_start = self.bag_last_person_nearby.get(tid, self.bag_first_seen[tid])
            if now - effective_start > self.abandoned_timeout_s:
                emit("ABANDONED_OBJECT", EventSeverity.SUSPICIOUS)

        # ---- Weapon events (HIGH_ALERT) ----
        if counts.get("knife", 0) > 0:
            emit("KNIFE_DETECTED", EventSeverity.HIGH_ALERT)

        if counts.get("pistol", 0) > 0:
            emit("PISTOL_DETECTED", EventSeverity.HIGH_ALERT)

        if counts.get("rifle", 0) > 0 or counts.get("assault_rifle", 0) > 0:
            emit("RIFLE_DETECTED", EventSeverity.HIGH_ALERT)

        # ---- Advance state ----
        self.prev_counts = dict(counts)
        return events

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_emit(
        self,
        event_type: str,
        severity: EventSeverity,
        counts: dict[str, int],
        tracked: list[dict],
        avg_conf: float,
        now: float,
    ) -> Event | None:
        cooldown = (
            self.high_alert_cooldown_s
            if severity == EventSeverity.HIGH_ALERT
            else self.event_cooldown_s
        )
        if now - self._last_event_times.get(event_type, 0.0) < cooldown:
            return None
        self._last_event_times[event_type] = now
        return Event(
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            event_type=event_type,
            severity=severity,
            timestamp=now,
            detections=dict(counts),
            tracks=[dict(t) for t in tracked],
            avg_confidence=avg_conf,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _avg_confidence(tracked: list[dict]) -> float:
    if not tracked:
        return 0.0
    return sum(obj.get("confidence", 0.0) for obj in tracked) / len(tracked)
