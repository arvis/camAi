"""llm/factory.py — Build the configured LLMProvider from environment variables."""

from __future__ import annotations

import os

from llm.base import LLMProvider, VisionProvider


def get_provider() -> LLMProvider:
    """Return an LLMProvider instance based on the LLM_PROVIDER env var.

    LLM_PROVIDER=openai    → OpenAIProvider (default)
    LLM_PROVIDER=anthropic → AnthropicProvider
    LLM_PROVIDER=gemini    → GeminiProvider (also implements VisionProvider)
    LLM_PROVIDER=ollama    → OllamaProvider
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    if provider == "gemini":
        from llm.gemini_provider import GeminiProvider
        return GeminiProvider()

    if provider == "ollama":
        from llm.ollama_provider import OllamaProvider
        return OllamaProvider()

    # Default: openai
    from llm.openai_provider import OpenAIProvider
    return OpenAIProvider()


def get_vision_provider() -> VisionProvider | None:
    """Return a VisionProvider if the configured LLM supports image analysis, else None.

    openai  → OpenAIProvider (uses gpt-4o for vision)
    gemini  → GeminiProvider
    Others  → None (Anthropic and Ollama do not implement VisionProvider)
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider()

    if provider == "gemini":
        from llm.gemini_provider import GeminiProvider
        return GeminiProvider()

    return None
