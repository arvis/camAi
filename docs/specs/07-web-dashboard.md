# Feature 7: Web Dashboard (Command Room)

## What it does
Browser UI showing live annotated camera feeds, real-time alert log, and situation report panel.

## Steps
1. FastAPI serves annotated MJPEG streams: `GET /stream/{cam_id}`
2. FastAPI serves the HTML dashboard page: `GET /`
3. Browser opens a WebSocket connection: `WS /ws`
4. Server pushes alert and report events to browser as JSON over WebSocket
5. Browser JS appends alerts to the log and updates the report panel in real time

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Dashboard HTML page |
| `GET` | `/stream/{cam_id}` | MJPEG stream with detection overlays |
| `WS` | `/ws` | Real-time alerts and reports |

## WebSocket message format
```json
// Alert event
{
  "type": "alert",
  "camera_name": "Drone-1",
  "event_type": "CROWD_FORMED",
  "objects": {"person": 3},
  "timestamp": "14:05:32"
}

// Situation report
{
  "type": "report",
  "text": "3 personnel observed near checkpoint...",
  "timestamp": "14:06:00"
}
```

## Dashboard layout
```
┌──────────────────────────────────────────┐
│  TACTICAL VISION AI              [LIVE]  │
├─────────────┬─────────────┬─────────────┤
│  Drone-1    │  Checkpoint │  Drone-2    │
│  [stream]   │  [stream]   │  [stream]   │
├─────────────┴──────┬──────┴─────────────┤
│  ALERTS            │  SITUATION REPORT  │
│  14:05 CROWD       │  3 personnel near  │
│  14:04 VEHICLE     │  checkpoint...     │
└────────────────────┴────────────────────┘
```

Dark tactical theme. No build tooling — plain HTML + CSS + vanilla JS.

## Module
`api/routes.py` — FastAPI routes
`api/mjpeg.py` — MJPEG multipart stream generator
`api/templates/index.html` — frontend
