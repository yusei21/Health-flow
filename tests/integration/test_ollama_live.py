"""Live check against a local Ollama. Opt-in: HEALTHFLOW_RUN_OLLAMA_TESTS=1."""

import os

import pytest

from app.agents.intent_agent import IntentAgent
from app.core.config import Settings
from app.llm.ollama_provider import OllamaLLMProvider
from app.schemas.symptoms import Symptom

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.anyio,
    pytest.mark.skipif(
        os.environ.get("HEALTHFLOW_RUN_OLLAMA_TESTS") != "1",
        reason="set HEALTHFLOW_RUN_OLLAMA_TESTS=1 with Ollama running",
    ),
]


async def test_extracts_red_flag_symptoms_from_portuguese_text() -> None:
    agent = IntentAgent(OllamaLLMProvider.from_settings(Settings()))
    extraction = await agent.extract("estou com uma dor forte no peito e falta de ar há 20 minutos")
    assert {Symptom.CHEST_PAIN, Symptom.SHORTNESS_OF_BREATH} <= set(extraction.symptoms)
    assert extraction.duration_minutes == 20


async def test_colloquial_portuguese_with_typo() -> None:
    """Opt-in model evaluation; fails if the model loses key symptom mentions."""
    agent = IntentAgent(OllamaLLMProvider.from_settings(Settings()))
    extraction = await agent.extract(
        "to com dor de bariga e diarreia comecou hoje, nao teve acidente"
    )
    assert Symptom.ABDOMINAL_PAIN in extraction.symptoms
    assert Symptom.DIARRHEA in extraction.symptoms
