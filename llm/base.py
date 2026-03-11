"""llm/base.py — LLMProvider protocol shared by all provider implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, system: str, user: str) -> str:
        """Send a completion request and return the model's text response."""
        ...


@runtime_checkable
class VisionProvider(Protocol):
    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Analyze an image and return a text description."""
        ...
