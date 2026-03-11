# Feature 6: AI Situation Report

## What it does
Every 2 minutes, collects recent events, sends them to an LLM, and posts the AI-written summary to Slack.

When the **alert level is HIGH** (default), every `PERSON_DETECTED` event that carries a frame snapshot is additionally assessed by the vision LLM for military indicators. A **CRITICAL** alert is posted to Slack immediately if a military or armed person is detected.

## Steps
1. Run a background scheduler that fires every 120 seconds
2. Collect all events from the last 2 minutes across all cameras
3. If no events occurred, skip
4. **When `alert_level = high`**: for every `PERSON_DETECTED` event with a frame snapshot, call the vision provider to assess military appearance
   - If military/armed detected → post a `MILITARY_PERSON_DETECTED` CRITICAL alert to Slack immediately (does not wait for the scheduled report)
5. Format events + any visual assessment notes into a per-camera text block
6. Call the LLM with a lookout operator prompt
7. Post the generated report to `SLACK_REPORTING_URL`

## Alert levels

| Level | Vision check on persons | Military alert severity |
|-------|------------------------|------------------------|
| `high` (default) | Yes — every `PERSON_DETECTED` with a snapshot | `CRITICAL` 🔴 |
| `normal` | No | — |

Set via `config.yaml` `report.alert_level` or override in code via `SituationReporter(alert_level=AlertLevel.HIGH)`.

## Event severity levels

| Severity | Used for |
|----------|----------|
| `NORMAL` | Routine detections (person entered, vehicle arrived) |
| `SUSPICIOUS` | Loitering, running, abandoned object |
| `HIGH_ALERT` | Weapon detections (knife, pistol, rifle) |
| `CRITICAL` | Military/armed person confirmed by vision LLM |

## Config (`.env`)
```
SLACK_WEBHOOK_URL=...        # event alerts
SLACK_REPORTING_URL=...      # AI situation reports (falls back to SLACK_WEBHOOK_URL if unset)
LLM_PROVIDER=openai          # openai | anthropic | ollama
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-... # optional
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Config (`config.yaml`)
```yaml
report:
  interval_seconds: 120
  alert_level: high         # normal | high (default: high)
```

## Prompt sent to LLM

**System:**
```
You are a security operator on active lookout duty monitoring live camera feeds.
Write a short, clear situation report — 2-3 sentences maximum.
State what is happening, where, and whether it requires attention.
Be direct and specific. No filler, no hedging.
```

**User:**
```
Observations from the last 2 minutes:

[Drone-1] 14:03 — PERSON_DETECTED
[Drone-1] 14:04 — CROWD_FORMED
[Checkpoint] 14:04 — VEHICLE_ARRIVED
[Drone-2] — No activity

Visual assessments (GPT-4o):
[Drone-1] Person in military uniform, blue armband on left arm.
```

## Vision prompt (per-person, alert_level=high only)
```
You are a tactical analyst reviewing a surveillance frame.
In 1-2 sentences answer:
(1) Does this person appear military or armed (uniform, tactical vest, helmet, weapon)?
(2) Are any colored identification bands visible on arms or head
(e.g. red armband, blue armband, colored headband used as IFF markers)?
Be concise and factual. If uncertain, say so.
```

Military keywords that trigger a CRITICAL alert: `military`, `armed`, `tactical`, `rifle`, `pistol`, `gun`, `firearm`, `soldier`, `combatant`, `armband`, `iff`.

## LLM abstraction

All providers implement the same interface:
```python
class LLMProvider(Protocol):
    async def complete(self, system: str, user: str) -> str: ...
```

| Provider | Class | Default model |
|----------|-------|---------------|
| `openai` | `OpenAIProvider` | `gpt-4o-mini` |
| `anthropic` | `AnthropicProvider` | `claude-haiku-4-5-20251001` |
| `ollama` | `OllamaProvider` | configurable via `OLLAMA_MODEL` |

Switch provider by changing `LLM_PROVIDER` in `.env` — no code changes needed.

## Fallback
If LLM call fails → post raw event list to Slack with a note.
If vision call fails → skip that frame, do not send alert (fail safe, not fail open).

## Module
`core/intelligence.py` — `SituationReporter` class with `asyncio` scheduler
`core/events.py` — `AlertLevel` enum, `EventSeverity` enum (includes `CRITICAL`)
`llm/` — provider implementations
