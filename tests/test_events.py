"""Tests for core/events.py — EventDetector state machine.

TDD order: unhappy/failing paths first, then happy paths.
Time is injected via the `now` parameter to avoid real clock dependency.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracked(
    track_id: int | None,
    class_name: str,
    direction: str = "stationary",
    bbox: tuple = (100, 100, 200, 200),
    confidence: float = 0.9,
    velocity_px: float = 0.0,
) -> dict:
    return {
        "track_id": track_id,
        "class_name": class_name,
        "movement_direction": direction,
        "bbox": bbox,
        "confidence": confidence,
        "velocity_px": velocity_px,
    }


def _make_detector(**kwargs):
    """Create an EventDetector with sensible defaults (overridable via kwargs)."""
    from core.events import EventDetector
    defaults = dict(
        camera_id="cam1",
        camera_name="Test",
        crowd_threshold=3,
        event_cooldown_s=30.0,
        high_alert_cooldown_s=10.0,
        loitering_timeout_s=45.0,
        running_threshold_px=80.0,
        abandoned_timeout_s=60.0,
        abandoned_proximity_px=150.0,
    )
    defaults.update(kwargs)
    return EventDetector(**defaults)


# ---------------------------------------------------------------------------
# Unhappy paths
# ---------------------------------------------------------------------------

class TestEventDetectorUnhappy:

    def test_empty_counts_no_events(self):
        """No detections → no events fired."""
        det = _make_detector()
        events = det.process({}, [], now=1000.0)
        assert events == []

    def test_stable_counts_no_person_detected(self):
        """Same person count as previous → PERSON_DETECTED not re-fired."""
        det = _make_detector()
        det.process({"person": 1}, [], now=1000.0)
        events = det.process({"person": 1}, [], now=1001.0)
        types = [e.event_type for e in events]
        assert "PERSON_DETECTED" not in types

    def test_stable_vehicle_count_no_vehicle_arrived(self):
        """Car count stable → VEHICLE_ARRIVED not re-fired."""
        det = _make_detector()
        det.process({"car": 1}, [], now=1000.0)
        events = det.process({"car": 1}, [], now=1001.0)
        types = [e.event_type for e in events]
        assert "VEHICLE_ARRIVED" not in types

    def test_cooldown_suppresses_repeat_person_detected(self):
        """PERSON_DETECTED not re-fired within cooldown window."""
        det = _make_detector(event_cooldown_s=30.0)
        # First: 0 → 1 person → fires
        det.process({"person": 1}, [], now=1000.0)
        # Drop to 0 then back to 1 → would fire again, but cooldown active
        det.process({}, [], now=1010.0)
        events = det.process({"person": 1}, [], now=1015.0)
        types = [e.event_type for e in events]
        assert "PERSON_DETECTED" not in types

    def test_high_alert_cooldown_suppresses_repeat_knife(self):
        """KNIFE_DETECTED not re-fired within high_alert_cooldown window."""
        det = _make_detector(high_alert_cooldown_s=10.0)
        det.process({"knife": 1}, [], now=1000.0)
        events = det.process({"knife": 1}, [], now=1005.0)
        types = [e.event_type for e in events]
        assert "KNIFE_DETECTED" not in types

    def test_loitering_not_fired_before_timeout(self):
        """Loitering must not fire until loitering_timeout_s has elapsed."""
        det = _make_detector(loitering_timeout_s=45.0)
        obj = _make_tracked(track_id=1, class_name="person", direction="stationary")
        det.process({"person": 1}, [obj], now=1000.0)
        events = det.process({"person": 1}, [obj], now=1030.0)  # only 30s < 45s
        types = [e.event_type for e in events]
        assert "LOITERING" not in types

    def test_running_not_fired_below_threshold(self):
        """RUNNING must not fire if velocity_px <= running_threshold_px."""
        det = _make_detector(running_threshold_px=80.0)
        obj = _make_tracked(track_id=1, class_name="person",
                            direction="N", velocity_px=79.9)
        events = det.process({"person": 1}, [obj], now=1000.0)
        types = [e.event_type for e in events]
        assert "RUNNING" not in types

    def test_abandoned_not_fired_before_timeout(self):
        """ABANDONED_OBJECT must not fire until abandoned_timeout_s elapses."""
        det = _make_detector(abandoned_timeout_s=60.0, abandoned_proximity_px=150.0)
        bag = _make_tracked(track_id=10, class_name="backpack", bbox=(300, 300, 400, 400))
        det.process({"backpack": 1}, [bag], now=1000.0)
        events = det.process({"backpack": 1}, [bag], now=1050.0)  # 50s < 60s
        types = [e.event_type for e in events]
        assert "ABANDONED_OBJECT" not in types

    def test_abandoned_not_fired_when_person_nearby(self):
        """ABANDONED_OBJECT must not fire when a person centroid is within proximity."""
        det = _make_detector(abandoned_timeout_s=60.0, abandoned_proximity_px=150.0)
        # bag centroid ≈ (350, 350)
        bag = _make_tracked(track_id=10, class_name="backpack", bbox=(300, 300, 400, 400))
        # person centroid ≈ (360, 360) — 14px away, well within 150px
        person = _make_tracked(track_id=1, class_name="person", bbox=(310, 310, 410, 410))
        det.process({"backpack": 1, "person": 1}, [bag, person], now=1000.0)
        events = det.process({"backpack": 1, "person": 1}, [bag, person], now=1080.0)
        types = [e.event_type for e in events]
        assert "ABANDONED_OBJECT" not in types

    def test_loitering_reset_when_track_moves(self):
        """LOITERING timer resets when the track moves, then must wait full timeout again."""
        det = _make_detector(loitering_timeout_s=45.0)
        obj_stat = _make_tracked(track_id=1, class_name="person", direction="stationary")
        obj_moving = _make_tracked(track_id=1, class_name="person", direction="E")
        det.process({"person": 1}, [obj_stat], now=1000.0)
        det.process({"person": 1}, [obj_stat], now=1040.0)  # 40s stationary — not yet fired
        det.process({"person": 1}, [obj_moving], now=1041.0)  # moved → reset
        det.process({"person": 1}, [obj_stat], now=1042.0)   # stationary again
        events = det.process({"person": 1}, [obj_stat], now=1060.0)  # only 18s since reset
        types = [e.event_type for e in events]
        assert "LOITERING" not in types

    def test_area_cleared_not_fired_when_already_empty(self):
        """AREA_CLEARED must not fire if prev_counts was already empty."""
        det = _make_detector()
        events = det.process({}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "AREA_CLEARED" not in types

    def test_crowd_not_formed_below_threshold(self):
        """CROWD_FORMED must not fire when person count is below crowd_threshold."""
        det = _make_detector(crowd_threshold=3)
        events = det.process({"person": 2}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "CROWD_FORMED" not in types

    def test_person_moving_not_fired_for_stationary(self):
        """PERSON_MOVING must not fire for stationary or unknown direction."""
        det = _make_detector()
        obj = _make_tracked(track_id=1, class_name="person", direction="stationary")
        events = det.process({"person": 1}, [obj], now=1000.0)
        types = [e.event_type for e in events]
        assert "PERSON_MOVING" not in types

    def test_running_not_fired_for_non_person(self):
        """RUNNING must only fire for persons, not bags or vehicles."""
        det = _make_detector(running_threshold_px=80.0)
        obj = _make_tracked(track_id=5, class_name="backpack",
                            direction="E", velocity_px=200.0)
        events = det.process({"backpack": 1}, [obj], now=1000.0)
        types = [e.event_type for e in events]
        assert "RUNNING" not in types


# ---------------------------------------------------------------------------
# Happy paths — transition events (NORMAL)
# ---------------------------------------------------------------------------

class TestEventDetectorHappyNormal:

    def test_person_detected_on_first_appearance(self):
        """PERSON_DETECTED fires when person count goes 0 → 1."""
        det = _make_detector()
        events = det.process({"person": 1}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "PERSON_DETECTED" in types

    def test_person_detected_severity_normal(self):
        """PERSON_DETECTED has NORMAL severity."""
        from core.events import EventSeverity
        det = _make_detector()
        events = det.process({"person": 1}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "PERSON_DETECTED")
        assert ev.severity == EventSeverity.NORMAL

    def test_vehicle_arrived_car(self):
        """VEHICLE_ARRIVED fires when car count goes 0 → 1."""
        det = _make_detector()
        events = det.process({"car": 1}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "VEHICLE_ARRIVED" in types

    def test_vehicle_arrived_truck(self):
        """VEHICLE_ARRIVED fires when truck count goes 0 → 1."""
        det = _make_detector()
        events = det.process({"truck": 1}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "VEHICLE_ARRIVED" in types

    def test_vehicle_arrived_severity_normal(self):
        """VEHICLE_ARRIVED has NORMAL severity."""
        from core.events import EventSeverity
        det = _make_detector()
        events = det.process({"car": 1}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "VEHICLE_ARRIVED")
        assert ev.severity == EventSeverity.NORMAL

    def test_crowd_formed_at_threshold(self):
        """CROWD_FORMED fires when person count reaches crowd_threshold."""
        det = _make_detector(crowd_threshold=3)
        events = det.process({"person": 3}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "CROWD_FORMED" in types

    def test_crowd_formed_above_threshold(self):
        """CROWD_FORMED fires when person count exceeds crowd_threshold."""
        det = _make_detector(crowd_threshold=3)
        events = det.process({"person": 5}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "CROWD_FORMED" in types

    def test_crowd_formed_severity_normal(self):
        """CROWD_FORMED has NORMAL severity."""
        from core.events import EventSeverity
        det = _make_detector(crowd_threshold=3)
        events = det.process({"person": 3}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "CROWD_FORMED")
        assert ev.severity == EventSeverity.NORMAL

    def test_person_moving_fires_for_non_stationary_direction(self):
        """PERSON_MOVING fires when a tracked person has direction N/S/E/W."""
        det = _make_detector()
        obj = _make_tracked(track_id=1, class_name="person", direction="N")
        events = det.process({"person": 1}, [obj], now=1000.0)
        types = [e.event_type for e in events]
        assert "PERSON_MOVING" in types

    def test_person_moving_severity_normal(self):
        """PERSON_MOVING has NORMAL severity."""
        from core.events import EventSeverity
        det = _make_detector()
        obj = _make_tracked(track_id=1, class_name="person", direction="E")
        events = det.process({"person": 1}, [obj], now=1000.0)
        ev = next(e for e in events if e.event_type == "PERSON_MOVING")
        assert ev.severity == EventSeverity.NORMAL

    def test_area_cleared_fires_when_all_counts_drop_to_zero(self):
        """AREA_CLEARED fires when all counts become 0 after non-empty state."""
        det = _make_detector()
        det.process({"person": 1}, [], now=1000.0)
        events = det.process({}, [], now=1001.0)
        types = [e.event_type for e in events]
        assert "AREA_CLEARED" in types

    def test_area_cleared_severity_normal(self):
        """AREA_CLEARED has NORMAL severity."""
        from core.events import EventSeverity
        det = _make_detector()
        det.process({"car": 2}, [], now=1000.0)
        events = det.process({}, [], now=1001.0)
        ev = next(e for e in events if e.event_type == "AREA_CLEARED")
        assert ev.severity == EventSeverity.NORMAL

    def test_cooldown_expires_allows_refire(self):
        """After cooldown expires, the same event can fire again."""
        det = _make_detector(event_cooldown_s=30.0)
        det.process({"person": 1}, [], now=1000.0)  # fires
        det.process({}, [], now=1001.0)              # cleared
        events = det.process({"person": 1}, [], now=1035.0)  # 35s > 30s cooldown
        types = [e.event_type for e in events]
        assert "PERSON_DETECTED" in types


# ---------------------------------------------------------------------------
# Happy paths — suspicious events
# ---------------------------------------------------------------------------

class TestEventDetectorHappySuspicious:

    def test_loitering_fires_after_timeout(self):
        """LOITERING fires when a person track is stationary for > loitering_timeout_s."""
        det = _make_detector(loitering_timeout_s=45.0)
        obj = _make_tracked(track_id=1, class_name="person", direction="stationary")
        det.process({"person": 1}, [obj], now=1000.0)
        events = det.process({"person": 1}, [obj], now=1046.0)  # 46s > 45s
        types = [e.event_type for e in events]
        assert "LOITERING" in types

    def test_loitering_severity_suspicious(self):
        """LOITERING has SUSPICIOUS severity."""
        from core.events import EventSeverity
        det = _make_detector(loitering_timeout_s=45.0)
        obj = _make_tracked(track_id=1, class_name="person", direction="stationary")
        det.process({"person": 1}, [obj], now=1000.0)
        events = det.process({"person": 1}, [obj], now=1050.0)
        ev = next(e for e in events if e.event_type == "LOITERING")
        assert ev.severity == EventSeverity.SUSPICIOUS

    def test_running_fires_above_threshold(self):
        """RUNNING fires when person velocity_px > running_threshold_px."""
        det = _make_detector(running_threshold_px=80.0)
        obj = _make_tracked(track_id=1, class_name="person",
                            direction="E", velocity_px=81.0)
        events = det.process({"person": 1}, [obj], now=1000.0)
        types = [e.event_type for e in events]
        assert "RUNNING" in types

    def test_running_severity_suspicious(self):
        """RUNNING has SUSPICIOUS severity."""
        from core.events import EventSeverity
        det = _make_detector(running_threshold_px=80.0)
        obj = _make_tracked(track_id=1, class_name="person",
                            direction="N", velocity_px=100.0)
        events = det.process({"person": 1}, [obj], now=1000.0)
        ev = next(e for e in events if e.event_type == "RUNNING")
        assert ev.severity == EventSeverity.SUSPICIOUS

    def test_abandoned_object_fires_after_timeout_no_person(self):
        """ABANDONED_OBJECT fires when bag alone for > abandoned_timeout_s."""
        det = _make_detector(abandoned_timeout_s=60.0, abandoned_proximity_px=150.0)
        bag = _make_tracked(track_id=10, class_name="suitcase", bbox=(300, 300, 400, 400))
        det.process({"suitcase": 1}, [bag], now=1000.0)
        events = det.process({"suitcase": 1}, [bag], now=1065.0)  # 65s > 60s
        types = [e.event_type for e in events]
        assert "ABANDONED_OBJECT" in types

    def test_abandoned_object_severity_suspicious(self):
        """ABANDONED_OBJECT has SUSPICIOUS severity."""
        from core.events import EventSeverity
        det = _make_detector(abandoned_timeout_s=60.0, abandoned_proximity_px=150.0)
        bag = _make_tracked(track_id=10, class_name="handbag", bbox=(300, 300, 400, 400))
        det.process({"handbag": 1}, [bag], now=1000.0)
        events = det.process({"handbag": 1}, [bag], now=1070.0)
        ev = next(e for e in events if e.event_type == "ABANDONED_OBJECT")
        assert ev.severity == EventSeverity.SUSPICIOUS

    def test_abandoned_fires_after_person_moves_away(self):
        """Timer resets when person nearby; ABANDONED fires timeout after person left."""
        det = _make_detector(abandoned_timeout_s=60.0, abandoned_proximity_px=150.0)
        bag = _make_tracked(track_id=10, class_name="backpack", bbox=(300, 300, 400, 400))
        # Person nearby at t=1000
        person = _make_tracked(track_id=1, class_name="person", bbox=(310, 310, 410, 410))
        det.process({"backpack": 1, "person": 1}, [bag, person], now=1000.0)
        # Person leaves at t=1050; bag alone from here
        det.process({"backpack": 1}, [bag], now=1050.0)
        # 65s after person left: 1050+65=1115 > 60s since person was nearby
        events = det.process({"backpack": 1}, [bag], now=1115.0)
        types = [e.event_type for e in events]
        assert "ABANDONED_OBJECT" in types

    def test_abandoned_all_bag_classes_trigger(self):
        """backpack, handbag, suitcase all trigger ABANDONED_OBJECT."""
        for cls in ("backpack", "handbag", "suitcase"):
            det = _make_detector(abandoned_timeout_s=60.0, abandoned_proximity_px=150.0)
            bag = _make_tracked(track_id=10, class_name=cls, bbox=(300, 300, 400, 400))
            det.process({cls: 1}, [bag], now=1000.0)
            events = det.process({cls: 1}, [bag], now=1070.0)
            types = [e.event_type for e in events]
            assert "ABANDONED_OBJECT" in types, f"Expected ABANDONED_OBJECT for {cls}"


# ---------------------------------------------------------------------------
# Happy paths — HIGH_ALERT events (weapons)
# ---------------------------------------------------------------------------

class TestEventDetectorHappyHighAlert:

    def test_knife_detected_fires(self):
        """KNIFE_DETECTED fires when knife appears in counts."""
        det = _make_detector()
        events = det.process({"knife": 1}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "KNIFE_DETECTED" in types

    def test_knife_detected_severity_high_alert(self):
        """KNIFE_DETECTED has HIGH_ALERT severity."""
        from core.events import EventSeverity
        det = _make_detector()
        events = det.process({"knife": 1}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "KNIFE_DETECTED")
        assert ev.severity == EventSeverity.HIGH_ALERT

    def test_pistol_detected_fires(self):
        """PISTOL_DETECTED fires when pistol appears in counts."""
        det = _make_detector()
        events = det.process({"pistol": 1}, [], now=1000.0)
        types = [e.event_type for e in events]
        assert "PISTOL_DETECTED" in types

    def test_pistol_detected_severity_high_alert(self):
        """PISTOL_DETECTED has HIGH_ALERT severity."""
        from core.events import EventSeverity
        det = _make_detector()
        events = det.process({"pistol": 1}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "PISTOL_DETECTED")
        assert ev.severity == EventSeverity.HIGH_ALERT

    def test_rifle_detected_fires(self):
        """RIFLE_DETECTED fires when rifle or assault_rifle appears in counts."""
        for cls in ("rifle", "assault_rifle"):
            det = _make_detector()
            events = det.process({cls: 1}, [], now=1000.0)
            types = [e.event_type for e in events]
            assert "RIFLE_DETECTED" in types, f"Expected RIFLE_DETECTED for class {cls}"

    def test_rifle_detected_severity_high_alert(self):
        """RIFLE_DETECTED has HIGH_ALERT severity."""
        from core.events import EventSeverity
        det = _make_detector()
        events = det.process({"rifle": 1}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "RIFLE_DETECTED")
        assert ev.severity == EventSeverity.HIGH_ALERT

    def test_high_alert_cooldown_expires_allows_refire(self):
        """KNIFE_DETECTED can re-fire after high_alert_cooldown_s elapses."""
        det = _make_detector(high_alert_cooldown_s=10.0)
        det.process({"knife": 1}, [], now=1000.0)
        events = det.process({"knife": 1}, [], now=1015.0)  # 15s > 10s cooldown
        types = [e.event_type for e in events]
        assert "KNIFE_DETECTED" in types


# ---------------------------------------------------------------------------
# Event data structure
# ---------------------------------------------------------------------------

class TestEventStructure:

    def test_event_has_required_fields(self):
        """Event objects have all required fields."""
        det = _make_detector()
        events = det.process({"person": 1}, [], now=1000.0)
        assert len(events) > 0
        ev = events[0]
        assert ev.camera_id == "cam1"
        assert ev.camera_name == "Test"
        assert ev.event_type
        assert ev.severity is not None
        assert ev.timestamp == 1000.0
        assert isinstance(ev.detections, dict)
        assert isinstance(ev.tracks, list)
        assert isinstance(ev.avg_confidence, float)

    def test_avg_confidence_from_tracked_objects(self):
        """avg_confidence is mean of confidence values in tracked list."""
        det = _make_detector()
        objs = [
            _make_tracked(track_id=1, class_name="person", confidence=0.8),
            _make_tracked(track_id=2, class_name="person", confidence=0.6),
        ]
        events = det.process({"person": 2}, objs, now=1000.0)
        assert len(events) > 0
        ev = next(e for e in events if e.event_type == "PERSON_DETECTED")
        assert abs(ev.avg_confidence - 0.7) < 1e-6

    def test_avg_confidence_zero_when_no_tracked(self):
        """avg_confidence is 0.0 when tracked list is empty."""
        det = _make_detector()
        events = det.process({"person": 1}, [], now=1000.0)
        ev = next(e for e in events if e.event_type == "PERSON_DETECTED")
        assert ev.avg_confidence == 0.0
