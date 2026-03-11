"""core/alerts.py — Slack (and Mattermost-compatible) alert/report sender.

AlertSender.send_alert(event)  — posts a formatted alert for a fired Event
AlertSender.send_report(text)  — posts a situation report to the reporting URL

Mattermost migration: change only SLACK_WEBHOOK_URL / SLACK_REPORTING_URL in
.env — payload is compatible.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import httpx

from core.events import Event, EventSeverity

logger = logging.getLogger(__name__)


class AlertSender:
    """Posts alerts and reports to Slack incoming webhooks.

    Parameters
    ----------
    webhook_url : str | None
        Webhook URL for event alerts. If None, alerts are no-ops.
    report_webhook_url : str | None
        Separate webhook URL for situation reports. Falls back to webhook_url
        when None.
    cooldown_s : float
        Minimum seconds between alerts for the same (camera_id, event_type).
    no_cooldown_types : set[str] | None
        Event types that bypass rate limiting (e.g. ``{"PERSON_MOVING"}``).
    """

    def __init__(
        self,
        webhook_url: str | None,
        report_webhook_url: str | None = None,
        cooldown_s: float = 30.0,
        no_cooldown_types: set[str] | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.report_webhook_url = report_webhook_url
        self.cooldown_s = cooldown_s
        self.no_cooldown_types: set[str] = no_cooldown_types or set()
        self._last_sent: dict[tuple[str, str], float] = {}

    @classmethod
    def from_env(cls) -> "AlertSender":
        """Construct from environment variables."""
        return cls(
            webhook_url=os.environ.get("SLACK_WEBHOOK_URL") or None,
            report_webhook_url=os.environ.get("SLACK_REPORTING_URL") or None,
            no_cooldown_types={"PERSON_MOVING"},
        )

    async def send_alert(self, event: Event) -> None:
        """Post a formatted alert for *event* to Slack.

        Rate-limited per (camera_id, event_type), unless the event type is in
        no_cooldown_types. No-op if webhook_url is None.
        """
        if not self.webhook_url:
            return

        if event.event_type not in self.no_cooldown_types:
            key = (event.camera_id, event.event_type)
            now = time.monotonic()
            if now - self._last_sent.get(key, 0.0) < self.cooldown_s:
                return
            self._last_sent[key] = now

        text = _format_alert(event)
        await self._post({"text": text}, url=self.webhook_url)

    async def send_report(self, text: str) -> None:
        """Post a situation report. Uses report_webhook_url if set, else webhook_url."""
        url = self.report_webhook_url or self.webhook_url
        if not url:
            return
        await self._post({"text": text}, url=url)

    async def _post(self, payload: dict, url: str) -> None:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Failed to post to Slack webhook: %s", exc)


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

_SEVERITY_LABEL = {
    EventSeverity.NORMAL: "[INFO]",
    EventSeverity.SUSPICIOUS: "[SUSPICIOUS]",
    EventSeverity.HIGH_ALERT: "[HIGH ALERT]",
    EventSeverity.CRITICAL: "[CRITICAL]",
}


def _format_alert(event: Event) -> str:
    label = _SEVERITY_LABEL.get(event.severity, "[ALERT]")
    ts = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).strftime("%H:%M:%S")

    objects_str = ", ".join(
        f"{count} {cls}" for cls, count in event.detections.items() if count > 0
    ) or "—"

    tracks_str = " · ".join(
        f"#{t['track_id']} {t.get('movement_direction', '?')}"
        for t in event.tracks
        if t.get("track_id") is not None
    ) or "—"

    conf_pct = f"{round(event.avg_confidence * 100)}%"

    return (
        f"{label} *ALERT — {event.camera_name}*\n"
        f"Event:      {event.event_type}\n"
        f"Objects:    {objects_str}\n"
        f"Tracks:     {tracks_str}\n"
        f"Confidence: {conf_pct}\n"
        f"Time:       {ts}"
    )
