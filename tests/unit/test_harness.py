import asyncio
import logging
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from app.core.exceptions import (
    FacilityProviderError,
    HarnessLimitError,
    HarnessPolicyError,
    HarnessTimeoutError,
    LLMUnavailableError,
)
from app.harness.actions import FinishReason, HarnessAction, Intent, PlannedAction, ReasonCode
from app.harness.autonomous_harness import AutonomousHealthFlowHarness
from app.harness.planner import DeterministicPlanner
from app.harness.policy import HarnessLimits, HarnessPolicy
from app.harness.response import build_routing_response
from app.harness.state import HarnessState, combine_assessments
from app.llm.schemas import ChatMessage
from app.ml.classifier import RoutingClassifier
from app.ml.inference import RoutingInferenceService
from app.safety.schemas import SafetyAssessment
from app.schemas.care import CareLevel, ServiceType
from app.schemas.facility import FacilityMatch
from app.schemas.patient import PatientContext
from app.schemas.routing import MLPrediction
from app.schemas.symptoms import SymptomExtraction
from app.tools.facilities import MockFacilityProvider
from tests.conftest import DEMO_USER_ID, SAO_PAULO, ScriptedLLMProvider, build_harness

pytestmark = pytest.mark.anyio

A = HarnessAction

RED_FLAG_PAYLOAD = {
    "symptoms": ["chest_pain", "shortness_of_breath"],
    "duration_minutes": 20,
    "severity": "severe",
}
MILD_PAYLOAD = {"symptoms": ["cough", "runny_nose"], "duration_minutes": 5760, "severity": "mild"}
CHEST_PAIN_ONLY: dict[str, object] = {"symptoms": ["chest_pain"], "severity": "moderate"}
RAW_TEXT_RED_FLAG = "ele teve uma convulsão agora"

FULL_PATH = [
    A.SAFETY_PRECHECK,
    A.EXTRACT_SYMPTOMS,
    A.LOAD_PATIENT_CONTEXT,
    A.RUN_SAFETY_ASSESSMENT,
    A.RUN_ML,
    A.APPLY_ROUTING,
    A.SEARCH_FACILITIES,
    A.FINALIZE,
]
PRECHECK_EMERGENCY_PATH = [A.SAFETY_PRECHECK, A.APPLY_ROUTING, A.SEARCH_FACILITIES, A.FINALIZE]


class SpyPlanner:
    """Plays a scripted prefix, then delegates; keeps the state for post-abort checks."""

    def __init__(self, script: Sequence[HarnessAction] = ()) -> None:
        self.script = list(script)
        self.state: HarnessState | None = None
        self._inner = DeterministicPlanner()

    def next_action(self, state: HarnessState) -> PlannedAction:
        self.state = state
        if self.script:
            return PlannedAction(self.script.pop(0), "TEST_SCRIPT")
        return self._inner.next_action(state)


class FixedInference(RoutingInferenceService):
    def __init__(self, prediction: MLPrediction) -> None:
        super().__init__(None)
        self._prediction = prediction

    def predict(self, extraction: SymptomExtraction, context: PatientContext) -> MLPrediction:
        return self._prediction


class SlowLLM(ScriptedLLMProvider):
    async def generate_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], output_model: type[T]
    ) -> T:
        await asyncio.sleep(1)
        return await super().generate_structured(messages, output_model)


class FailingFacilities:
    async def find_nearby(
        self, service_type: ServiceType, latitude: float, longitude: float, radius_km: float
    ) -> list[FacilityMatch]:
        raise FacilityProviderError("provider down")


async def run(harness: AutonomousHealthFlowHarness, message: str = "relato") -> HarnessState:
    return await harness.run("req-1", DEMO_USER_ID, message, *SAO_PAULO)


def spied_state(planner: SpyPlanner) -> HarnessState:
    assert planner.state is not None
    return planner.state


# --- normal and skipping paths -------------------------------------------------------


async def test_normal_flow_runs_every_action_in_order(classifier: RoutingClassifier) -> None:
    state = await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier))

    assert state.completed_actions == FULL_PATH
    assert state.finished and state.finish_reason is FinishReason.COMPLETED
    assert (state.step_count, state.llm_call_count, state.tool_call_count) == (8, 1, 3)
    assert state.ml_prediction is not None
    assert state.routing_decision and state.routing_decision.safety_override is False
    expected = state.routing_decision.service_type
    assert state.facilities
    assert all(m.facility.service_type is expected for m in state.facilities)


async def test_action_history_is_complete_and_auditable(classifier: RoutingClassifier) -> None:
    state = await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier))

    assert [r.step for r in state.action_history] == list(range(1, 9))
    assert [r.action for r in state.action_history] == FULL_PATH
    assert [r.reason_code for r in state.action_history] == [
        ReasonCode.START_WITH_SAFETY_PRECHECK,
        ReasonCode.NO_PRECHECK_RED_FLAG,
        ReasonCode.SYMPTOMS_EXTRACTED,
        ReasonCode.CONTEXT_LOADED,
        ReasonCode.NO_RED_FLAG_RUN_ML,
        ReasonCode.ML_STEP_DONE,
        ReasonCode.ROUTING_DECIDED,
        ReasonCode.FACILITY_SEARCH_DONE,
    ]
    assert all(r.status == "ok" and r.duration_ms >= 0 for r in state.action_history)


async def test_precheck_red_flag_skips_llm_and_ml(classifier: RoutingClassifier) -> None:
    llm = ScriptedLLMProvider(MILD_PAYLOAD)
    state = await run(build_harness(llm, classifier), RAW_TEXT_RED_FLAG)

    assert llm.calls == []  # did not wait for the LLM
    assert state.completed_actions == PRECHECK_EMERGENCY_PATH
    assert state.extracted_symptoms is None and state.ml_prediction is None
    assert state.routing_decision and state.routing_decision.care_level is CareLevel.EMERGENCY
    assert "SAFETY_RULE:RED_FLAG_003" in state.routing_decision.reason_codes
    assert state.facilities[0].facility.service_type is ServiceType.EMERGENCY_ROOM
    assert state.llm_call_count == 0


async def test_red_flag_found_after_extraction_skips_ml(classifier: RoutingClassifier) -> None:
    state = await run(build_harness(ScriptedLLMProvider(RED_FLAG_PAYLOAD), classifier))

    assert A.RUN_ML not in state.completed_actions
    assert state.completed_actions[-3:] == [A.APPLY_ROUTING, A.SEARCH_FACILITIES, A.FINALIZE]
    assert state.ml_prediction is None
    assert state.routing_decision and state.routing_decision.care_level is CareLevel.EMERGENCY
    assert state.action_history[4].reason_code == ReasonCode.RED_FLAG_SKIP_ML


# --- preserved degradation behavior --------------------------------------------------


async def test_ml_unavailable_uses_conservative_fallback() -> None:
    state = await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier=None))

    assert state.routing_decision and state.routing_decision.care_level is CareLevel.URGENT_CARE
    assert "ML_UNAVAILABLE_CONSERVATIVE_FALLBACK" in state.routing_decision.reason_codes
    assert [e.stage for e in state.errors] == [A.RUN_ML]
    assert state.action_history[4].status == "handled_error:MLInferenceError"
    assert state.finish_reason is FinishReason.COMPLETED


async def test_llm_unavailable_without_red_flag_raises_controlled_error(
    classifier: RoutingClassifier,
) -> None:
    planner = SpyPlanner()
    harness = build_harness(
        ScriptedLLMProvider(error=LLMUnavailableError("down")), classifier, planner=planner
    )
    with pytest.raises(LLMUnavailableError):
        await run(harness, "estou com tosse")

    state = spied_state(planner)
    assert state.completed_actions == [A.SAFETY_PRECHECK, A.EXTRACT_SYMPTOMS, A.FINALIZE]
    assert state.finish_reason is FinishReason.LLM_UNAVAILABLE
    assert state.routing_decision is None and state.ml_prediction is None


async def test_llm_unavailable_with_red_flag_text_still_routes_to_emergency(
    classifier: RoutingClassifier,
) -> None:
    llm = ScriptedLLMProvider(error=LLMUnavailableError("down"))
    state = await run(build_harness(llm, classifier), "minha mãe desmaiou")

    assert llm.calls == []  # the precheck decided before the LLM was needed
    assert state.routing_decision and state.routing_decision.care_level is CareLevel.EMERGENCY
    assert state.errors == []


async def test_facility_provider_failure_keeps_routing_decision(
    classifier: RoutingClassifier,
) -> None:
    harness = build_harness(
        ScriptedLLMProvider(RED_FLAG_PAYLOAD), classifier, facilities=FailingFacilities()
    )
    state = await run(harness)

    assert state.routing_decision and state.routing_decision.care_level is CareLevel.EMERGENCY
    assert [e.stage for e in state.errors] == [A.SEARCH_FACILITIES]
    assert state.finish_reason is FinishReason.COMPLETED
    response = build_routing_response(state)
    assert response.facility is None
    assert response.emergency_guidance and "192" in response.emergency_guidance


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
    assert "não substitui avaliação profissional" in response.disclaimer
    assert "192" in response.disclaimer


# --- safety floor --------------------------------------------------------------------


async def test_confident_ml_cannot_lower_the_safety_floor() -> None:
    confident_primary = MLPrediction(
        predicted_class=CareLevel.PRIMARY_CARE,
        confidence=0.99,
        probabilities={
            CareLevel.PRIMARY_CARE: 0.99,
            CareLevel.URGENT_CARE: 0.005,
            CareLevel.EMERGENCY: 0.005,
        },
        model_version="test",
    )
    harness = build_harness(
        ScriptedLLMProvider(CHEST_PAIN_ONLY),
        classifier=None,
        inference=FixedInference(confident_primary),
    )
    state = await run(harness)

    assert state.ml_prediction == confident_primary
    assert state.routing_decision and state.routing_decision.care_level is CareLevel.URGENT_CARE
    assert state.routing_decision.safety_override is True
    assert "SAFETY_RULE:CAUTION_001" in state.routing_decision.reason_codes


def test_combined_assessment_never_lowers_a_floor() -> None:
    precheck = SafetyAssessment(
        has_red_flag=True,
        matched_rules=["RED_FLAG_002"],
        suggested_minimum_care_level=CareLevel.EMERGENCY,
    )
    later = SafetyAssessment(
        has_red_flag=False, matched_rules=[], suggested_minimum_care_level=None
    )
    combined = combine_assessments(precheck, later)
    assert combined == precheck
    assert combine_assessments(None, None) is None


# --- policy guard --------------------------------------------------------------------


async def test_planner_cannot_run_ml_before_safety(classifier: RoutingClassifier) -> None:
    planner = SpyPlanner([A.RUN_ML])
    with pytest.raises(HarnessPolicyError, match="after the full safety assessment"):
        await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier, planner=planner))
    state = spied_state(planner)
    assert state.step_count == 0 and state.completed_actions == []
    assert state.finish_reason is FinishReason.POLICY_VIOLATION


async def test_planner_cannot_call_llm_after_precheck_red_flag(
    classifier: RoutingClassifier,
) -> None:
    llm = ScriptedLLMProvider(MILD_PAYLOAD)
    planner = SpyPlanner([A.SAFETY_PRECHECK, A.EXTRACT_SYMPTOMS])
    with pytest.raises(HarnessPolicyError, match="must not wait for the LLM"):
        await run(build_harness(llm, classifier, planner=planner), RAW_TEXT_RED_FLAG)
    assert llm.calls == []


async def test_planner_cannot_run_ml_on_red_flag(classifier: RoutingClassifier) -> None:
    script = [
        A.SAFETY_PRECHECK,
        A.EXTRACT_SYMPTOMS,
        A.LOAD_PATIENT_CONTEXT,
        A.RUN_SAFETY_ASSESSMENT,
        A.RUN_ML,
    ]
    planner = SpyPlanner(script)
    harness = build_harness(ScriptedLLMProvider(RED_FLAG_PAYLOAD), classifier, planner=planner)
    with pytest.raises(HarnessPolicyError, match="red flag: ML must not run"):
        await run(harness)
    assert spied_state(planner).ml_prediction is None


async def test_planner_cannot_route_before_full_safety_assessment(
    classifier: RoutingClassifier,
) -> None:
    planner = SpyPlanner([A.SAFETY_PRECHECK, A.APPLY_ROUTING])
    with pytest.raises(HarnessPolicyError, match="full safety assessment"):
        await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier, planner=planner))
    assert spied_state(planner).routing_decision is None


async def test_planner_cannot_skip_ml_without_red_flag(classifier: RoutingClassifier) -> None:
    script = [A.SAFETY_PRECHECK, A.EXTRACT_SYMPTOMS, A.LOAD_PATIENT_CONTEXT]
    planner = SpyPlanner([*script, A.RUN_SAFETY_ASSESSMENT, A.APPLY_ROUTING])
    with pytest.raises(HarnessPolicyError, match="ML step must be attempted"):
        await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier, planner=planner))


async def test_repeating_planner_cannot_loop(classifier: RoutingClassifier) -> None:
    planner = SpyPlanner([A.SAFETY_PRECHECK] * 50)
    with pytest.raises(HarnessPolicyError, match="already executed"):
        await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier, planner=planner))
    assert spied_state(planner).step_count == 1


def test_unsupported_intent_cannot_execute_any_action() -> None:
    state = HarnessState(
        request_id="r",
        user_id=DEMO_USER_ID,
        user_message="x",
        latitude=0,
        longitude=0,
        intent=Intent.MEDICATION,
    )
    with pytest.raises(HarnessPolicyError, match="not allowed for intent"):
        HarnessPolicy().validate(PlannedAction(A.SAFETY_PRECHECK, "TEST"), state)


async def test_finished_run_refuses_further_actions(classifier: RoutingClassifier) -> None:
    state = await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier))
    with pytest.raises(HarnessPolicyError, match="already finished"):
        HarnessPolicy().validate(PlannedAction(A.SEARCH_FACILITIES, "TEST"), state)


# --- termination guards --------------------------------------------------------------


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (HarnessLimits(max_steps=3), FinishReason.STEP_LIMIT),
        (HarnessLimits(max_llm_calls=0), FinishReason.LLM_CALL_LIMIT),
        (HarnessLimits(max_tool_calls=1), FinishReason.TOOL_CALL_LIMIT),
    ],
)
async def test_budgets_stop_the_run(
    classifier: RoutingClassifier, limits: HarnessLimits, reason: FinishReason
) -> None:
    planner = SpyPlanner()
    harness = build_harness(
        ScriptedLLMProvider(MILD_PAYLOAD),
        classifier,
        planner=planner,
        policy=HarnessPolicy(limits),
    )
    with pytest.raises(HarnessLimitError):
        await run(harness)
    state = spied_state(planner)
    assert state.finished and state.finish_reason is reason
    assert state.step_count <= limits.max_steps
    assert state.llm_call_count <= limits.max_llm_calls
    assert state.tool_call_count <= limits.max_tool_calls
    assert state.routing_decision is None


async def test_global_timeout_stops_a_hanging_run(classifier: RoutingClassifier) -> None:
    planner = SpyPlanner()
    harness = build_harness(
        SlowLLM(MILD_PAYLOAD),
        classifier,
        planner=planner,
        policy=HarnessPolicy(HarnessLimits(timeout_seconds=0.05)),
    )
    with pytest.raises(HarnessTimeoutError):
        await run(harness)
    assert spied_state(planner).finish_reason is FinishReason.TIMEOUT


async def test_default_limits_fit_the_longest_legitimate_path() -> None:
    limits = HarnessLimits()
    assert limits.max_steps >= len(FULL_PATH)
    assert limits.max_llm_calls >= 1 and limits.max_tool_calls >= 3


# --- audit logging -------------------------------------------------------------------


async def test_planner_decisions_are_logged_without_clinical_text(
    classifier: RoutingClassifier, caplog: pytest.LogCaptureFixture
) -> None:
    message = "tosse seca e nariz escorrendo há quatro dias, sou diabético"
    with caplog.at_level(logging.INFO):
        await run(build_harness(ScriptedLLMProvider(MILD_PAYLOAD), classifier), message)

    decisions = [r for r in caplog.records if r.getMessage() == "planner_decision"]
    assert len(decisions) == len(FULL_PATH)
    first = vars(decisions[0])
    assert first["request_id"] == "req-1"
    assert first["next_action"] == A.SAFETY_PRECHECK and first["current_action"] is None
    assert first["reason_code"] == ReasonCode.START_WITH_SAFETY_PRECHECK
    assert first["step_count"] == 0
    for record in caplog.records:
        rendered = str(vars(record))
        assert message not in rendered and "diabético" not in rendered
