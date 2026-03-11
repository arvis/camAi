"""Tests for llm/gemini_provider.py — Google Gemini provider.

TDD order: unhappy/edge cases first, then happy paths.
google-genai is mocked via sys.modules so tests run without the package installed.
"""

from __future__ import annotations

import sys
import importlib
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Helpers — build mock google.genai hierarchy
# ---------------------------------------------------------------------------

def _build_genai_mocks(response_text: str = "Situation report."):
    """Return (mock_genai_module, mock_types_module, mock_aio_models) all wired up."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

    mock_aio = MagicMock()
    mock_aio.models = mock_aio_models

    mock_client_instance = MagicMock()
    mock_client_instance.aio = mock_aio

    mock_client_cls = MagicMock(return_value=mock_client_instance)

    mock_genai = MagicMock()
    mock_genai.Client = mock_client_cls

    mock_types = MagicMock()
    mock_types.GenerateContentConfig = MagicMock(side_effect=lambda **kw: kw)
    mock_types.Part = MagicMock()
    mock_types.Part.from_bytes = MagicMock(
        side_effect=lambda data, mime_type: {"data": data, "mime": mime_type}
    )

    return mock_genai, mock_types, mock_aio_models


def _install_mocks(mock_genai, mock_types):
    """Inject mock modules into sys.modules and return cleanup function."""
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    mock_genai.types = mock_types  # ensure `from google.genai import types` gets our mock

    sys.modules["google"] = mock_google
    sys.modules["google.genai"] = mock_genai
    sys.modules["google.genai.types"] = mock_types

    # Force re-import of provider with fresh mocks
    sys.modules.pop("llm.gemini_provider", None)

    def cleanup():
        for key in ("google", "google.genai", "google.genai.types"):
            sys.modules.pop(key, None)
        sys.modules.pop("llm.gemini_provider", None)

    return cleanup


def _make_provider(mock_genai, mock_types, api_key: str = "fake-key"):
    cleanup = _install_mocks(mock_genai, mock_types)
    from llm.gemini_provider import GeminiProvider
    provider = GeminiProvider(api_key=api_key)
    return provider, cleanup


# ---------------------------------------------------------------------------
# Unhappy / edge-case tests (first, per TDD discipline)
# ---------------------------------------------------------------------------

def test_missing_api_key_raises(monkeypatch):
    """GeminiProvider must raise if GEMINI_API_KEY is not set and no api_key given."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mock_genai, mock_types, _ = _build_genai_mocks()
    cleanup = _install_mocks(mock_genai, mock_types)
    try:
        from llm.gemini_provider import GeminiProvider
        with pytest.raises((KeyError, ValueError)):
            GeminiProvider()
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_complete_api_failure_propagates(monkeypatch):
    """If the Gemini API raises, complete() must propagate the exception."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_genai, mock_types, mock_aio_models = _build_genai_mocks()
    mock_aio_models.generate_content = AsyncMock(side_effect=Exception("API error"))

    provider, cleanup = _make_provider(mock_genai, mock_types)
    try:
        with pytest.raises(Exception, match="API error"):
            await provider.complete(system="sys", user="user msg")
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_analyze_image_api_failure_propagates(monkeypatch):
    """If the Gemini vision API raises, analyze_image() must propagate."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_genai, mock_types, mock_aio_models = _build_genai_mocks()
    mock_aio_models.generate_content = AsyncMock(side_effect=Exception("Vision error"))

    provider, cleanup = _make_provider(mock_genai, mock_types)
    try:
        with pytest.raises(Exception, match="Vision error"):
            await provider.analyze_image(b"\xff\xd8fake", "describe this")
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_returns_text(monkeypatch):
    """complete() returns the text from the Gemini response."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_genai, mock_types, _ = _build_genai_mocks("Two persons spotted near the fence.")

    provider, cleanup = _make_provider(mock_genai, mock_types)
    try:
        result = await provider.complete(system="You are a lookout.", user="What do you see?")
    finally:
        cleanup()

    assert result == "Two persons spotted near the fence."


@pytest.mark.asyncio
async def test_complete_passes_system_instruction(monkeypatch):
    """complete() must call generate_content with system_instruction in config."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_genai, mock_types, mock_aio_models = _build_genai_mocks()

    provider, cleanup = _make_provider(mock_genai, mock_types)
    try:
        await provider.complete(system="You are a security operator.", user="Report.")
    finally:
        cleanup()

    mock_aio_models.generate_content.assert_called_once()
    call_kwargs = mock_aio_models.generate_content.call_args[1]
    config = call_kwargs.get("config")
    assert config is not None
    assert config.get("system_instruction") == "You are a security operator."


@pytest.mark.asyncio
async def test_analyze_image_returns_text(monkeypatch):
    """analyze_image() returns the text from the Gemini vision response."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_genai, mock_types, _ = _build_genai_mocks("Person in military uniform, red armband on right arm.")

    provider, cleanup = _make_provider(mock_genai, mock_types)
    try:
        result = await provider.analyze_image(b"\xff\xd8fake_jpeg", "Tactical assessment?")
    finally:
        cleanup()

    assert "military uniform" in result or "red armband" in result


@pytest.mark.asyncio
async def test_analyze_image_sends_image_bytes(monkeypatch):
    """analyze_image() must pass the image bytes as the first content part."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_genai, mock_types, mock_aio_models = _build_genai_mocks()

    fake_frame = b"\xff\xd8\xff\xe0specific_frame"
    provider, cleanup = _make_provider(mock_genai, mock_types)
    try:
        await provider.analyze_image(fake_frame, "Describe.")
    finally:
        cleanup()

    mock_aio_models.generate_content.assert_called_once()
    call_kwargs = mock_aio_models.generate_content.call_args[1]
    contents = call_kwargs.get("contents")
    assert contents is not None and len(contents) == 2
    image_part = contents[0]  # Part.from_bytes mock returns {"data": ..., "mime": ...}
    assert image_part["data"] == fake_frame


@pytest.mark.asyncio
async def test_env_api_key_used_when_no_explicit_key(monkeypatch):
    """GeminiProvider reads GEMINI_API_KEY from env when api_key not passed."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-123")
    mock_genai, mock_types, _ = _build_genai_mocks()
    cleanup = _install_mocks(mock_genai, mock_types)
    try:
        from llm.gemini_provider import GeminiProvider
        GeminiProvider()  # no explicit api_key
        mock_genai.Client.assert_called_once_with(api_key="env-key-123")
    finally:
        cleanup()
