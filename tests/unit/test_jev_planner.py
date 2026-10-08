from uuid import UUID

import pytest

from app.core.exceptions import JevProviderError
from app.harness.actions import HarnessAction
from app.harness.jev import JevDecision, JevPlanner, JevPlannerState
from app.harness.planner import DeterministicPlanner
from app.harness.policy import HarnessPolicy
from app.harness.state import HarnessState
from app.safety.schemas import SafetyAssessment
from app.schemas.care import CareLevel, ServiceType
from app.schemas.patient import PatientContext
from app.schemas.routing import RoutingDecision
from app.schemas.symptoms import SymptomExtraction

pytestmark = pytest.mark.anyio

A = HarnessAction
USER_ID = UUID("7d6f0f3e-2b8a-4c1e-9a52-3f4b8c2d1e90")


class FakeJevProvider:
    def __init__(
        self,
        decision: JevDecision | None = None,
        error: JevProviderError | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[JevPlannerState] = []

    async def choose_next_action(
        self,
        *,
        state: JevPlannerState,
        request_id: str,
        step_count: int,
        allowed_actions: tuple[HarnessAction, ...],
    ) -> JevDecision:
        self.calls.append(state)
        assert request_id == "req-secret"
        assert step_count == 6
        assert set(allowed_actions) == {A.SEARCH_FACILITIES, A.FINALIZE}
        if self.error is not None:
            raise self.error
        assert self.decision is not None
        return self.decision


def routing_ready_state() -> HarnessState:
    state = HarnessState(
        request_id="req-secret",
        user_id=USER_ID,
        user_message="texto clínico que nunca deve sair",
        latitude=-23.55,
        longitude=-46.64,
    )
    state.safety_precheck = SafetyAssessment(
        has_red_flag=False,
        matched_rules=[],
        suggested_minimum_care_level=None,
    )
    # These stages are marked as completed below; their outputs must exist.
    # Otherwise the deterministic fallback correctly treats extraction as failed.
    state.extracted_symptoms = SymptomExtraction()
    state.patient_context = PatientContext()
    state.safety_assessment = SafetyAssessment(
        has_red_flag=False,
        matched_rules=[],
        suggested_minimum_care_level=None,
    )
    state.routing_decision = RoutingDecision(
        care_level=CareLevel.PRIMARY_CARE,
        service_type=ServiceType.UBS,
        reason_codes=["TEST"],
        safety_override=False,
        ml_prediction=None,
    )
    state.completed_actions = [
        A.SAFETY_PRECHECK,
        A.EXTRACT_SYMPTOMS,
        A.LOAD_PATIENT_CONTEXT,
        A.RUN_SAFETY_ASSESSMENT,
        A.RUN_ML,
        A.APPLY_ROUTING,
    ]
    state.step_count = 6
    return state


async def test_jev_planner_accepts_high_probability_policy_allowed_action() -> None:
    provider = FakeJevProvider(
        JevDecision(A.FINALIZE, probability=0.92, model="jev-latest", latency_ms=12.0)
    )
    planner = JevPlanner(
        provider=provider,
        policy=HarnessPolicy(),
        min_probability=0.70,
        fallback=DeterministicPlanner(),
    )

    planned = await planner.next_action(routing_ready_state())

    assert planned.action is A.FINALIZE
    assert planned.reason_code == "JEV_PLANNER:finalize"
    assert len(provider.calls) == 1


async def test_low_probability_falls_back_to_deterministic_planner() -> None:
    provider = FakeJevProvider(
        JevDecision(A.FINALIZE, probability=0.42, model="jev-latest", latency_ms=12.0)
    )
    planner = JevPlanner(provider=provider, policy=HarnessPolicy(), min_probability=0.70)

    planned = await planner.next_action(routing_ready_state())

    assert planned.action is A.SEARCH_FACILITIES


async def test_disallowed_action_falls_back_to_deterministic_planner() -> None:
    provider = FakeJevProvider(
        JevDecision(A.RUN_ML, probability=0.99, model="jev-latest", latency_ms=12.0)
    )
    planner = JevPlanner(provider=provider, policy=HarnessPolicy(), min_probability=0.70)

    planned = await planner.next_action(routing_ready_state())

    assert planned.action is A.SEARCH_FACILITIES


async def test_provider_error_falls_back_to_deterministic_planner() -> None:
    provider = FakeJevProvider(error=JevProviderError("timeout"))
    planner = JevPlanner(provider=provider, policy=HarnessPolicy(), min_probability=0.70)

    planned = await planner.next_action(routing_ready_state())

    assert planned.action is A.SEARCH_FACILITIES


async def test_provider_boundary_receives_only_sanitized_state() -> None:
    provider = FakeJevProvider(
        JevDecision(A.FINALIZE, probability=0.95, model="jev-latest", latency_ms=10.0)
    )
    planner = JevPlanner(provider=provider, policy=HarnessPolicy(), min_probability=0.70)
    await planner.next_action(routing_ready_state())

    payload = provider.calls[0].model_dump()
    rendered = str(payload)
    assert "texto clínico" not in rendered
    assert "-23.55" not in rendered
    assert "-46.64" not in rendered
    assert str(USER_ID) not in rendered
    assert set(payload) == {
        "intent",
        "safety_precheck_complete",
        "has_red_flag",
        "symptoms_extracted",
        "patient_context_loaded",
        "safety_assessment_available",
        "ml_prediction_available",
        "routing_decision_available",
        "facility_results_available",
        "fatal_error_present",
        "completed_actions",
        "allowed_actions",
        "step_count",
    }
