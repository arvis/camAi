"""llm/anthropic_provider.py — Anthropic completion provider."""

from __future__ import annotations

import os


class AnthropicProvider:
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        message = await self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text if message.content else ""
