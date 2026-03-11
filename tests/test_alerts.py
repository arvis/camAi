"""Tests for core/alerts.py — Slack alert and report sending.

TDD order: unhappy/edge cases first, then happy paths.
httpx is mocked via pytest-mock to avoid real network calls.
"""

from __future__ import annotations

import time
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.events import Event, EventSeverity
from core.alerts import AlertSender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    camera_id: str = "cam1",
    camera_name: str = "Drone-1",
    event_type: str = "CROWD_FORMED",
    severity: EventSeverity = EventSeverity.NORMAL,
    timestamp: float = 1700000000.0,
    detections: dict | None = None,
    tracks: list | None = None,
    avg_confidence: float = 0.89,
) -> Event:
    return Event(
        camera_id=camera_id,
        camera_name=camera_name,
        event_type=event_type,
        severity=severity,
        timestamp=timestamp,
        detections=detections or {"person": 4},
        tracks=tracks or [
            {"track_id": 2, "movement_direction": "N"},
            {"track_id": 5, "movement_direction": "stationary"},
            {"track_id": 7, "movement_direction": "E"},
        ],
        avg_confidence=avg_confidence,
    )


# ---------------------------------------------------------------------------
# Unhappy / edge-case tests (first, per TDD discipline)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_alert_no_webhook_url_is_noop():
    """If SLACK_WEBHOOK_URL is not set, send_alert must not raise and must not make HTTP calls."""
    sender = AlertSender(webhook_url=None)
    with patch("httpx.AsyncClient") as mock_client_cls:
        await sender.send_alert(_make_event())
        mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_send_report_no_webhook_url_is_noop():
    """If SLACK_WEBHOOK_URL is not set, send_report must not raise."""
    sender = AlertSender(webhook_url=None)
    with patch("httpx.AsyncClient") as mock_client_cls:
        await sender.send_report("Situation normal.")
        mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_send_alert_http_error_does_not_raise():
    """HTTP error from Slack must be caught and not propagate."""
    import httpx
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )
    mock_post = AsyncMock(return_value=mock_response)
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        # Should not raise
        await sender.send_alert(_make_event())


@pytest.mark.asyncio
async def test_send_alert_rate_limited():
    """Same (camera_id, event_type) must not be sent twice within cooldown."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake", cooldown_s=30.0)
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    event = _make_event()
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event)
        await sender.send_alert(event)  # second call within cooldown

    assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_send_alert_different_cameras_not_rate_limited():
    """Different camera_ids for the same event_type are independent — both must fire."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake", cooldown_s=30.0)
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    event_a = _make_event(camera_id="cam1")
    event_b = _make_event(camera_id="cam2")
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event_a)
        await sender.send_alert(event_b)

    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_send_alert_different_event_types_not_rate_limited():
    """Different event_types on the same camera are independent — both must fire."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake", cooldown_s=30.0)
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    event_a = _make_event(event_type="CROWD_FORMED")
    event_b = _make_event(event_type="RUNNING")
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event_a)
        await sender.send_alert(event_b)

    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_send_alert_after_cooldown_fires_again():
    """After cooldown expires, the same event must fire again."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake", cooldown_s=1.0)
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    event = _make_event()
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event)
        # Manually expire the cooldown
        key = (event.camera_id, event.event_type)
        sender._last_sent[key] = time.monotonic() - 2.0
        await sender.send_alert(event)

    assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_alert_posts_to_webhook():
    """send_alert posts JSON to the configured webhook URL."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake")
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(_make_event())

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[0][0] == "https://hooks.slack.com/fake"


@pytest.mark.asyncio
async def test_send_alert_payload_contains_key_fields():
    """Payload text must contain camera name, event type, confidence, and object counts."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake")
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    event = _make_event(
        camera_name="Drone-1",
        event_type="CROWD_FORMED",
        detections={"person": 4},
        avg_confidence=0.89,
    )
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event)

    text = captured["json"]["text"]
    assert "Drone-1" in text
    assert "CROWD_FORMED" in text
    assert "89%" in text or "89" in text
    assert "person" in text


@pytest.mark.asyncio
async def test_send_alert_payload_contains_track_ids():
    """Payload must list track IDs."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake")
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    event = _make_event(tracks=[
        {"track_id": 2, "movement_direction": "N"},
        {"track_id": 5, "movement_direction": "stationary"},
    ])
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event)

    text = captured["json"]["text"]
    assert "#2" in text
    assert "#5" in text


@pytest.mark.asyncio
async def test_send_report_posts_text():
    """send_report posts a plain text payload to the webhook."""
    sender = AlertSender(webhook_url="https://hooks.slack.com/fake")
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["json"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_report("All clear. No suspicious activity.")

    assert captured["json"]["text"] == "All clear. No suspicious activity."


@pytest.mark.asyncio
async def test_make_from_env_reads_webhook_url(monkeypatch):
    """AlertSender.from_env() reads SLACK_WEBHOOK_URL from environment."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/env-test")
    sender = AlertSender.from_env()
    assert sender.webhook_url == "https://hooks.slack.com/env-test"


@pytest.mark.asyncio
async def test_make_from_env_missing_returns_none_url(monkeypatch):
    """AlertSender.from_env() with no env var sets webhook_url to None."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    sender = AlertSender.from_env()
    assert sender.webhook_url is None


# ---------------------------------------------------------------------------
# Reporting URL tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_report_uses_report_webhook_url():
    """send_report posts to report_webhook_url when it is set."""
    sender = AlertSender(
        webhook_url="https://hooks.slack.com/alerts",
        report_webhook_url="https://hooks.slack.com/reports",
    )
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_report("Situation normal.")

    assert captured["url"] == "https://hooks.slack.com/reports"


@pytest.mark.asyncio
async def test_send_report_falls_back_to_webhook_url_when_no_report_url():
    """send_report falls back to webhook_url when report_webhook_url is None."""
    sender = AlertSender(
        webhook_url="https://hooks.slack.com/alerts",
        report_webhook_url=None,
    )
    captured = {}

    async def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = fake_post

    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_report("Situation normal.")

    assert captured["url"] == "https://hooks.slack.com/alerts"


@pytest.mark.asyncio
async def test_from_env_reads_report_webhook_url(monkeypatch):
    """from_env() reads SLACK_REPORTING_URL into report_webhook_url."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/alerts")
    monkeypatch.setenv("SLACK_REPORTING_URL", "https://hooks.slack.com/reports")
    sender = AlertSender.from_env()
    assert sender.report_webhook_url == "https://hooks.slack.com/reports"


@pytest.mark.asyncio
async def test_from_env_report_url_missing_is_none(monkeypatch):
    """from_env() sets report_webhook_url to None when SLACK_REPORTING_URL is absent."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/alerts")
    monkeypatch.delenv("SLACK_REPORTING_URL", raising=False)
    sender = AlertSender.from_env()
    assert sender.report_webhook_url is None


# ---------------------------------------------------------------------------
# no_cooldown_types tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_cooldown_type_fires_every_time():
    """Event types in no_cooldown_types must not be rate-limited."""
    sender = AlertSender(
        webhook_url="https://hooks.slack.com/fake",
        cooldown_s=30.0,
        no_cooldown_types={"PERSON_MOVING"},
    )
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    event = _make_event(event_type="PERSON_MOVING")
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event)
        await sender.send_alert(event)
        await sender.send_alert(event)

    assert mock_post.call_count == 3


@pytest.mark.asyncio
async def test_normal_type_still_rate_limited_when_no_cooldown_types_set():
    """Other event types remain rate-limited even when no_cooldown_types is configured."""
    sender = AlertSender(
        webhook_url="https://hooks.slack.com/fake",
        cooldown_s=30.0,
        no_cooldown_types={"PERSON_MOVING"},
    )
    mock_post = AsyncMock()
    mock_post.return_value = MagicMock(raise_for_status=MagicMock())
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    event = _make_event(event_type="CROWD_FORMED")
    with patch("httpx.AsyncClient", return_value=mock_client):
        await sender.send_alert(event)
        await sender.send_alert(event)

    assert mock_post.call_count == 1
