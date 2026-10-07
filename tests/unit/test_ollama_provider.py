"""Exercises the real OpenAI SDK client against a scripted HTTP transport."""

import json
from collections.abc import Callable

import httpx2 as httpx  # transport library used by the openai SDK
import pytest
from openai import AsyncOpenAI

from app.agents.intent_agent import LLMSymptomOutput
from app.core.exceptions import LLMResponseError, LLMUnavailableError
from app.llm.ollama_provider import OllamaLLMProvider
from app.llm.schemas import ChatMessage, Role
from app.schemas.symptoms import Severity, Symptom

pytestmark = pytest.mark.anyio
MESSAGES = [ChatMessage(role=Role.USER, content="dor no peito")]


def completion(content: str) -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def provider(
    handler: Callable[[httpx.Request], httpx.Response], validation_retries: int = 1
) -> OllamaLLMProvider:
    client = AsyncOpenAI(
        base_url="http://llm.test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OllamaLLMProvider(client, "qwen3:4b", max_validation_retries=validation_retries)


async def test_parses_valid_structured_output_and_requests_json_schema() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = '{"symptoms": ["chest_pain"], "duration_minutes": 20, "severity": "severe"}'
        return httpx.Response(200, json=completion(body))

    result = await provider(handler).generate_structured(MESSAGES, LLMSymptomOutput)

    assert result.symptoms == [Symptom.CHEST_PAIN]
    assert result.severity is Severity.SEVERE
    assert seen[0]["response_format"]["type"] == "json_schema"  # type: ignore[index]
    assert seen[0]["temperature"] == 0


async def test_strips_thinking_block_before_validation() -> None:
    body = '<think>hmm</think>\n{"symptoms": [], "severity": "mild"}'
    result = await provider(
        lambda _: httpx.Response(200, json=completion(body))
    ).generate_structured(MESSAGES, LLMSymptomOutput)
    assert result.severity is Severity.MILD


async def test_drops_symptom_codes_outside_vocabulary() -> None:
    body = '{"symptoms": ["chest_pain", "heart_attack"], "severity": "unknown"}'
    result = await provider(
        lambda _: httpx.Response(200, json=completion(body))
    ).generate_structured(MESSAGES, LLMSymptomOutput)
    assert result.symptoms == [Symptom.CHEST_PAIN]


async def test_retries_invalid_json_then_succeeds() -> None:
    bodies = iter(["not json at all", '{"symptoms": [], "severity": "mild"}'])
    result = await provider(
        lambda _: httpx.Response(200, json=completion(next(bodies)))
    ).generate_structured(MESSAGES, LLMSymptomOutput)
    assert result.severity is Severity.MILD


async def test_gives_up_after_bounded_retries() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion("{broken"))

    with pytest.raises(LLMResponseError):
        await provider(handler, validation_retries=2).generate_structured(
            MESSAGES, LLMSymptomOutput
        )
    assert calls == 3


async def test_connection_failure_becomes_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailableError):
        await provider(handler).generate_structured(MESSAGES, LLMSymptomOutput)


async def test_timeout_becomes_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMUnavailableError):
        await provider(handler).generate_structured(MESSAGES, LLMSymptomOutput)


async def test_http_error_becomes_llm_unavailable() -> None:
    with pytest.raises(LLMUnavailableError):
        await provider(lambda _: httpx.Response(500, json={"error": "boom"})).generate_structured(
            MESSAGES, LLMSymptomOutput
        )
