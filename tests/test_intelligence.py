"""Tests for core/intelligence.py — AI situation report generation.

TDD order: unhappy/edge cases first, then happy paths.
LLM provider and AlertSender are mocked to avoid real network calls.
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.events import AlertLevel, Event, EventSeverity
from core.intelligence import SituationReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    camera_id: str = "cam1",
    camera_name: str = "Drone-1",
    event_type: str = "PERSON_DETECTED",
    severity: EventSeverity = EventSeverity.NORMAL,
    timestamp: float | None = None,
    detections: dict | None = None,
    tracks: list | None = None,
    avg_confidence: float = 0.85,
    frame_snapshot: bytes | None = None,
) -> Event:
    return Event(
        camera_id=camera_id,
        camera_name=camera_name,
        event_type=event_type,
        severity=severity,
        timestamp=timestamp if timestamp is not None else time.time(),
        detections=detections or {"person": 1},
        tracks=tracks or [{"track_id": 1, "movement_direction": "N"}],
        avg_confidence=avg_confidence,
        frame_snapshot=frame_snapshot,
    )


def _make_vision_provider(response: str = "Person in military uniform, blue armband on left arm.") -> AsyncMock:
    vp = AsyncMock()
    vp.analyze_image = AsyncMock(return_value=response)
    return vp


def _make_llm(response: str = "Situation report text.") -> AsyncMock:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


def _make_sender() -> AsyncMock:
    sender = AsyncMock()
    sender.send_report = AsyncMock()
    sender.send_alert = AsyncMock()
    return sender


# ---------------------------------------------------------------------------
# Unhappy / edge-case tests (first, per TDD discipline)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_events_skips_llm_and_report():
    """If no events were recorded, _generate_and_post must not call LLM or send report."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    await reporter._generate_and_post()

    llm.complete.assert_not_called()
    sender.send_report.assert_not_called()


@pytest.mark.asyncio
async def test_llm_failure_posts_raw_event_list():
    """If LLM raises an exception, raw event list is posted with a fallback note."""
    llm = _make_llm()
    llm.complete = AsyncMock(side_effect=Exception("LLM unavailable"))
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event(event_type="CROWD_FORMED", camera_name="Drone-1"))

    await reporter._generate_and_post()

    sender.send_report.assert_called_once()
    text = sender.send_report.call_args[0][0]
    assert "CROWD_FORMED" in text or "Drone-1" in text
    # Should contain a fallback indicator
    assert "[" in text  # raw event list format


@pytest.mark.asyncio
async def test_events_cleared_after_report():
    """After _generate_and_post, the event buffer is cleared so next run starts fresh."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event())
    await reporter._generate_and_post()

    # Second call — buffer should be empty, so LLM not called again
    llm.complete.reset_mock()
    sender.send_report.reset_mock()
    await reporter._generate_and_post()

    llm.complete.assert_not_called()
    sender.send_report.assert_not_called()


@pytest.mark.asyncio
async def test_send_report_failure_does_not_raise():
    """If AlertSender.send_report raises, SituationReporter must not propagate."""
    llm = _make_llm()
    sender = _make_sender()
    sender.send_report = AsyncMock(side_effect=Exception("Network down"))
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event())
    # Must not raise
    await reporter._generate_and_post()


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_events_formatted_in_user_prompt():
    """LLM user prompt must include camera name, timestamp, and event type."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event(
        camera_name="Checkpoint",
        event_type="VEHICLE_ARRIVED",
        timestamp=1700000000.0,
    ))

    await reporter._generate_and_post()

    llm.complete.assert_called_once()
    call_kwargs = llm.complete.call_args
    # accept both positional and keyword args
    user_prompt = call_kwargs[1].get("user") or call_kwargs[0][1]
    assert "Checkpoint" in user_prompt
    assert "VEHICLE_ARRIVED" in user_prompt


@pytest.mark.asyncio
async def test_llm_receives_system_prompt():
    """LLM must be called with a tactical analyst system prompt."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event())
    await reporter._generate_and_post()

    call_kwargs = llm.complete.call_args
    system_prompt = call_kwargs[1].get("system") or call_kwargs[0][0]
    assert "security operator" in system_prompt.lower() or "lookout" in system_prompt.lower()


@pytest.mark.asyncio
async def test_llm_response_posted_as_report():
    """The LLM response text must be passed directly to send_report."""
    llm = _make_llm(response="Two persons moving north. Truck at checkpoint.")
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event())
    await reporter._generate_and_post()

    sender.send_report.assert_called_once_with(
        "Two persons moving north. Truck at checkpoint."
    )


@pytest.mark.asyncio
async def test_multiple_cameras_all_in_prompt():
    """Events from multiple cameras all appear in the LLM user prompt."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event(camera_name="Drone-1", event_type="PERSON_DETECTED"))
    reporter.record_event(_make_event(camera_id="cam2", camera_name="Checkpoint", event_type="VEHICLE_ARRIVED"))

    await reporter._generate_and_post()

    call_kwargs = llm.complete.call_args
    user_prompt = call_kwargs[1].get("user") or call_kwargs[0][1]
    assert "Drone-1" in user_prompt
    assert "Checkpoint" in user_prompt


@pytest.mark.asyncio
async def test_camera_with_no_events_shows_no_activity():
    """Cameras registered but with no events should appear as 'No activity'."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        interval_s=120.0,
        camera_names={"cam1": "Drone-1", "cam2": "Checkpoint"},
    )

    reporter.record_event(_make_event(camera_id="cam1", camera_name="Drone-1"))
    # cam2 has no events

    await reporter._generate_and_post()

    call_kwargs = llm.complete.call_args
    user_prompt = call_kwargs[1].get("user") or call_kwargs[0][1]
    assert "Checkpoint" in user_prompt
    assert "No activity" in user_prompt


@pytest.mark.asyncio
async def test_record_event_is_thread_safe():
    """record_event can be called from multiple threads without data corruption."""
    import threading
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    errors = []

    def add_events():
        try:
            for _ in range(50):
                reporter.record_event(_make_event())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_events) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(reporter._events) == 200


# ---------------------------------------------------------------------------
# Vision analysis tests (unhappy first, then happy)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_vision_provider_no_vision_calls():
    """Without a vision_provider, no analyze_image calls are made even with a frame."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    # LLM should still be called (text-only report)
    llm.complete.assert_called_once()
    # No vision provider — analyze_image never called
    call_kwargs = llm.complete.call_args
    user_prompt = call_kwargs[1].get("user") or call_kwargs[0][1]
    # No vision annotation injected
    assert "Visual" not in user_prompt


@pytest.mark.asyncio
async def test_vision_provider_failure_does_not_block_report():
    """If analyze_image raises, the report is still generated from events alone."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider()
    vp.analyze_image = AsyncMock(side_effect=Exception("Vision API down"))
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    # Must not raise
    await reporter._generate_and_post()
    llm.complete.assert_called_once()
    sender.send_report.assert_called_once()


@pytest.mark.asyncio
async def test_vision_only_called_for_person_detected_events():
    """analyze_image is only called for PERSON_DETECTED events, not others."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="VEHICLE_ARRIVED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    vp.analyze_image.assert_not_called()


@pytest.mark.asyncio
async def test_vision_analysis_included_in_llm_prompt():
    """When analyze_image succeeds, its result appears in the LLM user prompt."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Person in military uniform, blue armband on left arm.")
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        camera_name="Drone-1",
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    vp.analyze_image.assert_called_once()
    call_kwargs = llm.complete.call_args
    user_prompt = call_kwargs[1].get("user") or call_kwargs[0][1]
    assert "military uniform" in user_prompt or "blue armband" in user_prompt


@pytest.mark.asyncio
async def test_vision_not_called_for_person_event_without_snapshot():
    """If PERSON_DETECTED has no frame_snapshot, analyze_image is not called."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(event_type="PERSON_DETECTED", frame_snapshot=None))

    await reporter._generate_and_post()

    vp.analyze_image.assert_not_called()


@pytest.mark.asyncio
async def test_vision_called_with_correct_frame_bytes():
    """analyze_image receives the exact frame_snapshot bytes from the event."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    fake_frame = b"\xff\xd8\xff\xe0specific_frame_data"
    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=fake_frame,
    ))

    await reporter._generate_and_post()

    vp.analyze_image.assert_called_once()
    called_bytes = vp.analyze_image.call_args[0][0]
    assert called_bytes == fake_frame


# ---------------------------------------------------------------------------
# Military HIGH_ALERT tests (unhappy first, then happy)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_military_alert_not_sent_without_vision_provider():
    """No vision provider → send_alert is never called even with a frame snapshot."""
    llm = _make_llm()
    sender = _make_sender()
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_military_alert_not_sent_without_frame_snapshot():
    """PERSON_DETECTED with no frame_snapshot → analyze_image not called → no alert."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Person in military uniform.")
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(event_type="PERSON_DETECTED", frame_snapshot=None))

    await reporter._generate_and_post()

    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_military_alert_not_sent_when_civilian():
    """Vision says civilian → send_alert not called."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Person appears to be a civilian in casual clothing. Nothing suspicious.")
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_military_alert_not_sent_when_vision_fails():
    """Vision API failure → no HIGH_ALERT sent (fail safe, not fail open)."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider()
    vp.analyze_image = AsyncMock(side_effect=Exception("Vision API down"))
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_military_high_alert_sent_when_military_detected():
    """Vision identifies military uniform → send_alert called with CRITICAL event when alert_level=HIGH."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Person in military uniform with tactical vest.")
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_called_once()
    alert_event = sender.send_alert.call_args[0][0]
    assert alert_event.severity.value == "CRITICAL"


@pytest.mark.asyncio
async def test_military_high_alert_event_type():
    """HIGH_ALERT event sent for military detection has event_type MILITARY_PERSON_DETECTED."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Armed soldier with rifle and helmet.")
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        camera_id="cam2",
        camera_name="Checkpoint",
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_called_once()
    alert_event = sender.send_alert.call_args[0][0]
    assert alert_event.event_type == "MILITARY_PERSON_DETECTED"
    assert alert_event.camera_id == "cam2"
    assert alert_event.camera_name == "Checkpoint"


@pytest.mark.asyncio
async def test_military_keywords_trigger_high_alert(request):
    """Each military keyword individually triggers a HIGH_ALERT."""
    keywords_and_phrases = [
        "military uniform",
        "armed with a pistol",
        "tactical gear visible",
        "carrying a firearm",
        "assault rifle visible",
        "appears to be a soldier",
    ]
    for phrase in keywords_and_phrases:
        llm = _make_llm()
        sender = _make_sender()
        vp = _make_vision_provider(phrase)
        reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

        reporter.record_event(_make_event(
            event_type="PERSON_DETECTED",
            frame_snapshot=b"\xff\xd8\xff\xe0fake",
        ))
        await reporter._generate_and_post()

        assert sender.send_alert.called, f"Expected CRITICAL alert for phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# Alert level tests (unhappy first, then happy)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vision_not_called_when_alert_level_normal():
    """At NORMAL alert level, vision analysis is skipped even with provider and snapshot."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Person in military uniform.")
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        interval_s=120.0,
        vision_provider=vp,
        alert_level=AlertLevel.NORMAL,
    )

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    vp.analyze_image.assert_not_called()
    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_military_alert_not_sent_when_alert_level_normal():
    """No CRITICAL alert at NORMAL alert level even if vision would detect military."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Armed soldier with tactical vest and rifle.")
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        interval_s=120.0,
        vision_provider=vp,
        alert_level=AlertLevel.NORMAL,
    )

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_vision_called_when_alert_level_high():
    """At HIGH alert level, vision analysis fires for PERSON_DETECTED with snapshot."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Person appears to be a civilian.")
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        interval_s=120.0,
        vision_provider=vp,
        alert_level=AlertLevel.HIGH,
    )

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    vp.analyze_image.assert_called_once()


@pytest.mark.asyncio
async def test_critical_alert_severity_on_military_detection():
    """Military detection at HIGH alert level sends CRITICAL (not HIGH_ALERT) severity."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Armed soldier in military uniform.")
    reporter = SituationReporter(
        llm=llm,
        alert_sender=sender,
        interval_s=120.0,
        vision_provider=vp,
        alert_level=AlertLevel.HIGH,
    )

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_called_once()
    alert_event = sender.send_alert.call_args[0][0]
    assert alert_event.severity == EventSeverity.CRITICAL


@pytest.mark.asyncio
async def test_negated_military_keywords_do_not_trigger_alert():
    """Vision response starting with NO must not send a CRITICAL alert.

    Regression: keyword matching on 'armed' in 'does not appear to be military
    or armed' was firing false-positive CRITICAL alerts.
    """
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider(
        "NO. The person does not appear to be military or armed; "
        "no uniform, tactical vest, helmet, or weapon is visible."
    )
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    sender.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_default_alert_level_is_high():
    """SituationReporter defaults to HIGH alert level without explicit parameter."""
    llm = _make_llm()
    sender = _make_sender()
    vp = _make_vision_provider("Armed soldier with rifle.")
    # No alert_level= passed — should default to HIGH
    reporter = SituationReporter(llm=llm, alert_sender=sender, interval_s=120.0, vision_provider=vp)

    reporter.record_event(_make_event(
        event_type="PERSON_DETECTED",
        frame_snapshot=b"\xff\xd8\xff\xe0fake_jpeg",
    ))

    await reporter._generate_and_post()

    # Default is HIGH → military detection → CRITICAL alert
    sender.send_alert.assert_called_once()
    alert_event = sender.send_alert.call_args[0][0]
    assert alert_event.severity == EventSeverity.CRITICAL
