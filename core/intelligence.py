"""core/intelligence.py — AI situation report scheduler.

SituationReporter collects Event objects from camera threads, then every
`interval_s` seconds calls the LLM to generate a tactical summary and posts
it to Slack via AlertSender.

If a VisionProvider is supplied, PERSON_DETECTED events that carry a
frame_snapshot are analyzed for military appearance and colored IFF armbands
(red/blue) before the situation report is generated.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone

from core.alerts import AlertSender
from core.events import AlertLevel, Event, EventSeverity
from llm.base import LLMProvider, VisionProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a security operator on active lookout duty monitoring live camera feeds. "
    "Write a short, clear situation report — 2-3 sentences maximum. "
    "State what is happening, where, and whether it requires attention. "
    "Be direct and specific. No filler, no hedging."
)

_VISION_PROMPT = (
    "You are a tactical analyst reviewing a surveillance frame. "
    "Answer YES if ANY of the following are visible on the person: "
    "helmet, tactical vest, military uniform, combat boots, weapon, firearm, "
    "ammunition pouches, or any other military/combat equipment. "
    "A single indicator is enough to answer YES. "
    "Answer NO only if none of these indicators are present. "
    "Start your response with YES or NO, then briefly state which indicator(s) you saw. "
    "Also note any colored IFF identification bands (red/blue armbands, headbands)."
)


class SituationReporter:
    """Periodically generates an AI situation report from recent events.

    Parameters
    ----------
    llm : LLMProvider
        Async LLM provider (OpenAI, Anthropic, or Ollama).
    alert_sender : AlertSender
        Used to post the report to Slack.
    interval_s : float
        Seconds between reports (default 120).
    camera_names : dict[str, str] | None
        Optional mapping of camera_id → camera_name used to show cameras with
        no activity in the prompt.
    vision_provider : VisionProvider | None
        Optional vision-capable provider. When set and alert_level is HIGH,
        PERSON_DETECTED events with a frame_snapshot are analyzed for military
        appearance and IFF armbands before the text report is generated.
    alert_level : AlertLevel
        Operational alert posture. When HIGH (default), every PERSON_DETECTED
        event with a frame snapshot is checked via the vision provider for
        military indicators — a CRITICAL alert is sent if detected. When
        NORMAL, vision checks are skipped.
    """

    def __init__(
        self,
        llm: LLMProvider,
        alert_sender: AlertSender,
        interval_s: float = 120.0,
        camera_names: dict[str, str] | None = None,
        vision_provider: VisionProvider | None = None,
        alert_level: AlertLevel = AlertLevel.HIGH,
    ) -> None:
        self._llm = llm
        self._sender = alert_sender
        self._interval_s = interval_s
        self._camera_names: dict[str, str] = dict(camera_names or {})
        self._vision = vision_provider
        self._alert_level = alert_level
        self._events: list[Event] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_event(self, event: Event) -> None:
        """Thread-safe: called from camera threads to log a fired event."""
        with self._lock:
            self._events.append(event)

    async def run(self) -> None:
        """Asyncio task — runs forever, fires a report every interval_s."""
        while True:
            await asyncio.sleep(self._interval_s)
            await self._generate_and_post()

    async def _send_military_alert(self, source_event: Event, analysis: str) -> None:
        """Fire an immediate CRITICAL alert when vision detects a military/armed person."""
        alert = Event(
            camera_id=source_event.camera_id,
            camera_name=source_event.camera_name,
            event_type="MILITARY_PERSON_DETECTED",
            severity=EventSeverity.CRITICAL,
            timestamp=time.time(),
            detections=source_event.detections,
            tracks=source_event.tracks,
            avg_confidence=source_event.avg_confidence,
            frame_snapshot=source_event.frame_snapshot,
        )
        try:
            await self._sender.send_alert(alert)
            logger.warning(
                "MILITARY_PERSON_DETECTED on %s — vision: %s",
                source_event.camera_name,
                analysis,
            )
        except Exception as exc:
            logger.warning("Failed to send military HIGH_ALERT for %s: %s", source_event.camera_name, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _generate_and_post(self) -> None:
        """Collect buffered events, call LLM, post report. Clears the buffer."""
        with self._lock:
            events = list(self._events)
            self._events.clear()

        if not events:
            return

        # Vision analysis: assess PERSON_DETECTED frames when provider available and alert level is HIGH
        visual_notes: list[tuple[str, str]] = []  # [(camera_name, analysis), ...]
        if self._vision is not None and self._alert_level == AlertLevel.HIGH:
            for ev in events:
                if ev.event_type == "PERSON_DETECTED" and ev.frame_snapshot:
                    try:
                        analysis = await self._vision.analyze_image(ev.frame_snapshot, _VISION_PROMPT)
                        if analysis:
                            visual_notes.append((ev.camera_name, analysis))
                            if _is_military(analysis):
                                await self._send_military_alert(ev, analysis)
                    except Exception as exc:
                        logger.warning("Vision analysis failed for %s: %s", ev.camera_name, exc)

        user_prompt = _build_user_prompt(events, self._camera_names, visual_notes)

        try:
            report = await self._llm.complete(system=_SYSTEM_PROMPT, user=user_prompt)
        except Exception as exc:
            logger.warning("LLM call failed (%s); posting raw event list.", exc)
            report = _build_fallback_report(events)

        try:
            await self._sender.send_report(report)
        except Exception as exc:
            logger.warning("Failed to post situation report: %s", exc)


# ---------------------------------------------------------------------------
# Prompt / fallback helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(
    events: list[Event],
    camera_names: dict[str, str],
    visual_notes: list[tuple[str, str]],
) -> str:
    """Format events grouped by camera for the LLM user message."""
    # Group events by camera
    by_camera: dict[str, list[Event]] = {}
    for ev in events:
        by_camera.setdefault(ev.camera_id, []).append(ev)

    lines = ["Observations from the last 2 minutes:\n"]

    # Cameras with events
    for cam_id, cam_events in by_camera.items():
        for ev in cam_events:
            ts = datetime.fromtimestamp(ev.timestamp, tz=timezone.utc).strftime("%H:%M")
            lines.append(f"[{ev.camera_name}] {ts} — {ev.event_type}")

    # Cameras registered with no events
    for cam_id, cam_name in camera_names.items():
        if cam_id not in by_camera:
            lines.append(f"[{cam_name}] — No activity")

    # Visual assessments from GPT-4o
    if visual_notes:
        lines.append("\nVisual assessments (GPT-4o):")
        for cam_name, analysis in visual_notes:
            lines.append(f"[{cam_name}] {analysis}")

    return "\n".join(lines)


_MILITARY_KEYWORDS = frozenset({
    "military",
    "armed",
    "tactical",
    "rifle",
    "pistol",
    "gun",
    "firearm",
    "soldier",
    "combatant",
    "armband",
    "iff",
})


def _is_military(analysis: str) -> bool:
    """Return True if the vision analysis affirmatively identifies military/armed indicators.

    Expects the LLM to start with YES or NO (per _VISION_PROMPT). Falls back to
    keyword matching for unstructured responses (e.g. test fixtures), but only
    when no YES/NO prefix is present.
    """
    stripped = analysis.strip().lower()
    if stripped.startswith("yes"):
        return True
    if stripped.startswith("no"):
        return False
    # Fallback for responses that don't follow the YES/NO format
    return any(kw in stripped for kw in _MILITARY_KEYWORDS)


def _build_fallback_report(events: list[Event]) -> str:
    lines = ["[LLM unavailable — raw event log]\n"]
    for ev in events:
        ts = datetime.fromtimestamp(ev.timestamp, tz=timezone.utc).strftime("%H:%M:%S")
        lines.append(f"[{ev.camera_name}] {ts} — {ev.event_type}")
    return "\n".join(lines)
