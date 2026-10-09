from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import ScriptedLLMProvider
from tests.integration.test_api import AUTH, make_client


@pytest.mark.parametrize(
    ("content", "mime", "expected"),
    [
        (b"", "audio/webm", 422),
        (b"not audio", "text/plain", 415),
        (b"x" * (10 * 1024 * 1024 + 1), "audio/webm", 413),
    ],
)
def test_invalid_audio_never_reaches_provider(content: bytes, mime: str, expected: int) -> None:
    with make_client(ScriptedLLMProvider(), None) as demo:
        app = create_app(
            Settings(transcription_url="https://transcriber.test/audio/transcriptions"),
            cast(FastAPI, demo.app).state.container,
        )
        with TestClient(app) as client, patch("app.api.routes.audio.httpx.AsyncClient") as provider:
            response = client.post(
                "/api/v1/audio/transcribe?consent=true",
                headers={**AUTH, "Content-Type": mime},
                content=content,
            )
            assert response.status_code == expected
            provider.assert_not_called()


def test_audio_returns_reviewable_text_not_a_routing_decision() -> None:
    with make_client(ScriptedLLMProvider(), None) as demo:
        app = create_app(
            Settings(transcription_url="https://transcriber.test/audio/transcriptions"),
            cast(FastAPI, demo.app).state.container,
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = httpx.Response(
            200,
            json={"text": "Estou com dor de cabeça"},
            request=httpx.Request("POST", "https://transcriber.test/audio/transcriptions"),
        )
        with TestClient(app) as client, patch("app.api.routes.audio.httpx.AsyncClient") as provider:
            provider.return_value.__aenter__.return_value = mock_client
            response = client.post(
                "/api/v1/audio/transcribe?consent=true",
                headers={**AUTH, "Content-Type": "audio/webm"},
                content=b"fake test audio",
            )
            assert response.status_code == 200
            assert response.json() == {"text": "Estou com dor de cabeça"}
            assert response.headers["cache-control"] == "no-store"
            assert mock_client.post.call_args.kwargs["data"]["language"] == "pt"
