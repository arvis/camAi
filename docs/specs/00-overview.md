# Tactical Vision AI — MVP Overview

## What we're building

Phones stream live video over local shared WiFi network → server detects objects and events → command room watches live annotated feeds on a dashboard → Slack gets alerts and AI situation reports.

## Pipeline

```
Phones (IP Webcam app, same shared WiFi network)
  → MJPEG streams → server reads frames per camera thread
    → YOLOv8 + ByteTrack: detect objects, assign track IDs
      → event detection: rules fire on state changes
        → dashboard: live annotated feeds + alert log + AI report (command room)
        → Slack webhook: alerts + situation reports (phone notifications)
```

## MVP Features

| # | File | Feature | Priority |
|---|------|---------|----------|
| 1 | [01-stream-ingestion.md](01-stream-ingestion.md) | Phone stream ingestion (IP Webcam MJPEG) | CORE |
| 2 | [02-object-detection.md](02-object-detection.md) | YOLOv8 object detection + overlay | CORE |
| 3 | [03-object-tracking.md](03-object-tracking.md) | ByteTrack object tracking | CORE |
| 4 | [04-event-detection.md](04-event-detection.md) | Event detection (rules-based) | CORE |
| 5 | [05-slack-alerts.md](05-slack-alerts.md) | Slack webhook alerts | CORE |
| 6 | [06-ai-situation-report.md](06-ai-situation-report.md) | AI situation report → Slack | CORE |
| 7 | [07-web-dashboard.md](07-web-dashboard.md) | Web dashboard (command room live view) | CORE |

## Out of MVP
- Cross-camera person correlation
- Threat scoring
- Map visualization
- Mattermost (swap Slack webhook URL when needed — API identical)

## File Structure

```
camAi/
├── main.py               # starts camera threads + FastAPI + report scheduler
├── config.yaml           # camera list + detection thresholds
├── .env                  # API keys, Slack webhook URL
├── requirements.txt
├── core/
│   ├── stream.py         # CameraStream — one thread per camera
│   ├── detector.py       # YOLOv8 + ByteTrack
│   ├── events.py         # event rules + state per camera
│   ├── alerts.py         # Slack webhook sender
│   └── intelligence.py   # LLM situation report scheduler
├── llm/
│   ├── base.py           # LLMProvider protocol + factory
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── ollama_provider.py
└── api/
    ├── routes.py         # FastAPI routes + WebSocket
    ├── mjpeg.py          # MJPEG stream generator
    └── templates/
        └── index.html    # dashboard frontend
```

## Dependencies

```
ultralytics              # YOLOv8 + ByteTrack
opencv-python-headless
fastapi
uvicorn
httpx                    # async HTTP (Slack webhook + LLM calls)
openai
anthropic
python-dotenv
pyyaml
```

## Slack → Mattermost migration
Change one line in `.env`:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
# later:
SLACK_WEBHOOK_URL=https://your-mattermost/hooks/...
```
No code changes needed.
