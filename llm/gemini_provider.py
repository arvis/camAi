"""llm/gemini_provider.py — Google Gemini completion and vision provider."""

from __future__ import annotations

import os


class GeminiProvider:
    """Implements both LLMProvider (text) and VisionProvider (image) via Gemini.

    Parameters
    ----------
    model : str
        Gemini model name. Default: gemini-2.0-flash (fast, supports vision).
    api_key : str | None
        Gemini API key. Falls back to GEMINI_API_KEY env var.
    """

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None) -> None:
        from google import genai  # lazy — only fails if google-genai not installed
        self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        """Send a text completion request and return the model's response."""
        from google.genai import types
        response = await self._client.aio.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=system),
            contents=user,
        )
        return response.text or ""

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Analyze a JPEG frame and return a tactical assessment."""
        from google.genai import types
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ]
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
        )
        return response.text or ""
