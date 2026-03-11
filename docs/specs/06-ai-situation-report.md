# Feature 6: AI Situation Report

## What it does
Every 2 minutes, collects recent events, sends them to an LLM, and posts the AI-written summary to Discord.

## Steps
1. Run a background scheduler that fires every 120 seconds
2. Collect all events from the last 2 minutes across all cameras
3. If no events occurred, skip
4. Format events into a per-camera text block
5. Call the LLM with a tactical analyst prompt
6. Post the generated report to Discord as a blue embed

## Config (`.env`)
```
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
```

## Prompt sent to LLM

**System:**
```
You are a tactical intelligence analyst receiving live camera feed data.
Write a concise situation report (3-5 sentences).
Focus on movement patterns, activity, and any threats.
Use direct, military-style language.
```

**User:**
```
Observations from the last 2 minutes:

[Drone-1] 14:03 — PERSON_DETECTED: 1 person moving N
[Drone-1] 14:04 — CROWD_FORMED: 3 persons
[Checkpoint] 14:04 — VEHICLE_ARRIVED: 1 truck
[Drone-2] — No activity
```

## LLM abstraction

All providers implement the same interface:
```python
class LLMProvider(Protocol):
    async def complete(self, prompt: str) -> str: ...
```

| Provider | Class | Default model |
|----------|-------|---------------|
| `openai` | `OpenAIProvider` | `gpt-4o-mini` |
| `anthropic` | `AnthropicProvider` | `claude-haiku-4-5` |
| `ollama` | `OllamaProvider` | configurable |

Switch provider by changing `LLM_PROVIDER` in `.env` — no code changes needed.

## Fallback
If LLM call fails → post raw event list to Discord with a note.

## Module
`core/intelligence.py` — `SituationReporter` class with `asyncio` scheduler
`llm/` — provider implementations
