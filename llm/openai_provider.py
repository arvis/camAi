"""llm/openai_provider.py — OpenAI completion and vision provider."""

from __future__ import annotations

import base64
import os


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self.model = model

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    async def analyze_image(self, image_bytes: bytes, prompt: str) -> str:
        """Send a JPEG frame to GPT-4o vision and return the tactical assessment."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = await self._client.chat.completions.create(
            model="gpt-4o",  # vision requires gpt-4o, not 4o-mini
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=150,
        )
        return response.choices[0].message.content or ""
