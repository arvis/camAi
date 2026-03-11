# Feature 5: Slack Alerts

## What it does
Sends a Slack message every time an event fires, and posts AI situation reports.
Uses incoming webhooks — no bot token or Slack app approval needed.

## Slack setup
1. Go to your Slack workspace → Apps → Incoming Webhooks
2. Enable → Add New Webhook → pick a channel (e.g. `#alerts`)
3. Copy the webhook URL → add to `.env`

## Steps
1. Read `SLACK_WEBHOOK_URL` from `.env`
2. Receive an `Event` from event detection
3. Build a Slack message payload with event details
4. POST to the webhook URL using `httpx` (async)

## Config (`.env`)
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
```

## Message format
```
🚨 *ALERT — Drone-1*
Event:      CROWD_FORMED
Objects:    4 persons
Tracks:     #2 N · #5 stationary · #7 E
Confidence: 89%
Time:       14:05:32
```

## Rate limit
Inherits `event_cooldown_s` from event detection — same event not sent twice within 30s per camera.

## Mattermost migration
Change only `SLACK_WEBHOOK_URL` in `.env` to your Mattermost incoming webhook URL.
The payload format is compatible — no code changes needed.

## Module
`core/alerts.py`
- `send_alert(event)` — posts an alert message
- `send_report(text)` — posts a situation report (called by Feature 6)
