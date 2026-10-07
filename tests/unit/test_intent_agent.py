import pytest

from app.agents.intent_agent import IntentAgent
from app.llm.schemas import Role
from app.schemas.symptoms import Severity, Symptom
from tests.conftest import ScriptedLLMProvider

pytestmark = pytest.mark.anyio


async def test_maps_llm_output_to_domain_extraction() -> None:
    llm = ScriptedLLMProvider(
        {
            "symptoms": ["chest_pain", "shortness_of_breath", "chest_pain"],
            "duration_minutes": 20,
            "severity": "severe",
            "age": None,
        }
    )
    extraction = await IntentAgent(llm).extract("dor forte no peito e falta de ar há 20 minutos")

    assert extraction.symptoms == [Symptom.CHEST_PAIN, Symptom.SHORTNESS_OF_BREATH]
    assert extraction.duration_minutes == 20
    assert extraction.severity is Severity.SEVERE


async def test_out_of_range_numbers_are_discarded_not_trusted() -> None:
    llm = ScriptedLLMProvider({"symptoms": [], "duration_minutes": -5, "age": 400})
    extraction = await IntentAgent(llm).extract("algo")
    assert extraction.duration_minutes is None
    assert extraction.age is None


async def test_prompt_forbids_diagnosis_and_sends_report_as_user_data_only() -> None:
    llm = ScriptedLLMProvider({"symptoms": []})
    await IntentAgent(llm).extract("relato do usuário")

    system, user = llm.calls[0]
    assert system.role is Role.SYSTEM
    assert "NÃO faça diagnóstico" in system.content
    assert user.role is Role.USER
    assert user.content == "relato do usuário"
