import pytest

from app.core.exceptions import LLMUnavailableError
from app.harness.healthflow_harness import HealthFlowHarness
from app.harness.response import build_routing_response
from app.harness.state import HarnessState, Stage
from app.ml.classifier import RoutingClassifier
from app.schemas.care import CareLevel, ServiceType
from app.tools.facilities import MockFacilityProvider
from tests.conftest import DEMO_USER_ID, SAO_PAULO, ScriptedLLMProvider, build_harness

pytestmark = pytest.mark.anyio

RED_FLAG_PAYLOAD = {
    "symptoms": ["chest_pain", "shortness_of_breath"],
    "duration_minutes": 20,
    "severity": "severe",
}
MILD_PAYLOAD = {"symptoms": ["cough", "runny_nose"], "duration_minutes": 5760, "severity": "mild"}


async def run(harness: HealthFlowHarness, message: str = "relato") -> HarnessState:
    return await harness.run("req-1", DEMO_USER_ID, message, *SAO_PAULO)


async def test_red_flag_skips_ml_and_routes_to_emergency(classifier: RoutingClassifier) -> None:
    state = await run(build_harness(ScriptedLLMProvider(RED_FLAG_PAYLOAD), classifier))

    assert state.safety_assessment and state.safety_assessment.has_red_flag
    assert state.ml_prediction is None
    assert state.routing_decision and state.routing_decision.care_level is CareLevel.EMERGENCY
    assert state.facilities[0].facility.service_type is ServiceType.EMERGENCY_ROOM


async def test_mild_report_uses_ml_and_finds_ubs(classifier: RoutingClassifier) -> None:
    state = await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier))

    assert state.ml_prediction is not None
    assert state.routing_decision and state.routing_decision.safety_override is False
    assert state.facilities
    expected = state.routing_decision.service_type
    assert all(m.facility.service_type is expected for m in state.facilities)


async def test_llm_unavailable_without_red_flag_raises_controlled_error(
    classifier: RoutingClassifier,
) -> None:
    harness = build_harness(ScriptedLLMProvider(error=LLMUnavailableError("down")), classifier)
    with pytest.raises(LLMUnavailableError):
        await run(harness, "estou com tosse")


async def test_llm_unavailable_with_red_flag_text_still_routes_to_emergency(
    classifier: RoutingClassifier,
) -> None:
    harness = build_harness(ScriptedLLMProvider(error=LLMUnavailableError("down")), classifier)
    state = await run(harness, "minha mãe desmaiou")

    assert state.extracted_symptoms is None
    assert state.routing_decision and state.routing_decision.care_level is CareLevel.EMERGENCY
    assert [e.stage for e in state.errors] == [Stage.EXTRACT_USER_REPORT]


async def test_ml_unavailable_uses_conservative_fallback() -> None:
    state = await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier=None))

    assert state.routing_decision and state.routing_decision.care_level is CareLevel.URGENT_CARE
    assert "ML_UNAVAILABLE_CONSERVATIVE_FALLBACK" in state.routing_decision.reason_codes
    assert [e.stage for e in state.errors] == [Stage.ML_CLASSIFIER]


async def test_no_facility_found_still_returns_guidance(classifier: RoutingClassifier) -> None:
    harness = build_harness(
        ScriptedLLMProvider(RED_FLAG_PAYLOAD), classifier, facilities=MockFacilityProvider([])
    )
    response = build_routing_response(await run(harness))

    assert response.facility is None
    assert "Nenhuma unidade compatível" in response.next_step
    assert response.emergency_guidance and "192" in response.emergency_guidance


async def test_response_never_contains_diagnosis_field(classifier: RoutingClassifier) -> None:
    response = build_routing_response(
        await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier))
    )
    assert "diagnosis" not in response.model_dump()
    assert "NÃO UTILIZAR PARA DECISÕES CLÍNICAS REAIS" in response.disclaimer
