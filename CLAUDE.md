# CLAUDE.md — Tactical Vision AI

## Project
Hackathon project: phones stream live video over shared WiFi network (e.g. Starlink router in the field) → server detects objects and events → command room watches live annotated feeds on dashboard → Slack gets alerts and AI situation reports.

## Docs
- [docs/plan.md](docs/plan.md) — original project plan
- [docs/specs/](docs/specs/) — one spec file per MVP feature (read before implementing a feature)

## Architecture
```
Phones (IP Webcam MJPEG, shared WiFi network)
  → CameraStream threads (core/stream.py) → frame buffer
    → detector (core/detector.py) — YOLOv8 + ByteTrack
      → events (core/events.py) — state machine per camera
        → alerts (core/alerts.py) — Slack webhook
        → intelligence (core/intelligence.py) — LLM scheduler → Slack
        → api/ — MJPEG streams + WebSocket → dashboard (command room)
```

## Testing discipline (mandatory)

- Write tests **before** writing implementation code (TDD)
- Write failing/unhappy path tests **first**, before happy path tests
- Code is only correct when all tests pass — do not mark a task done until tests are green
- Never write implementation to make a specific test pass by hardcoding; tests must reflect real behaviour

## Key rules

### LLM provider
- Default: OpenAI `gpt-4o-mini`
- Abstracted via `LLMProvider` protocol in `llm/base.py`
- Swappable via `.env` (`LLM_PROVIDER=openai|anthropic|ollama`) — never hardcode a provider
- When adding LLM calls, always go through the provider interface

### Config
- Camera sources and thresholds live in `config.yaml`
- Secrets (API keys, webhook URL) live in `.env`
- Never hardcode URLs, thresholds, or API keys in source files

### Concurrency
- One `threading.Thread` per camera in `core/stream.py`
- Background situation report scheduler runs as an `asyncio` task inside FastAPI lifespan
- FastAPI runs in the main thread via uvicorn

### Detection
- Model: `yolov8n.pt` (nano) — upgrade to `yolov8s.pt` only if GPU confirmed available
- Run detection every `DETECTION_INTERVAL` frames (config default: 3), not every frame
- ByteTrack is used via `model.track(..., tracker="bytetrack.yaml", persist=True)` — no separate tracking lib

### Alerts
- Slack incoming webhook POST via `httpx` (async)
- Rate limit: same `(camera_id, event_type)` not re-sent within `event_cooldown_s` seconds
- Alert and report sending both go through `core/alerts.py`
- Mattermost migration: change only `SLACK_WEBHOOK_URL` in `.env` — payload is compatible

### Dashboard
- Core feature (command room live view) — not a stretch goal
- Plain HTML + vanilla JS — no React, no build tooling
- MJPEG streams served from FastAPI (`/stream/{cam_id}`)
- WebSocket (`/ws`) for real-time alert/report push to browser

## What NOT to do
- Do not add object tracking libraries (DeepSORT, etc.) — ByteTrack only
- Do not build a React frontend unless explicitly asked
- Do not add cross-camera correlation logic — post-MVP
- Do not add threat scoring — post-MVP
- Do not use `cv2.imshow()` — headless server environment

## Testing without phones
Set `url: 0` (local webcam) or `url: ./test_footage/video.mp4` in `config.yaml` camera entries.
