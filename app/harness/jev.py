"""Optional Jev-backed planner for the autonomous harness.

Only a sanitized workflow state is sent to Jev. Clinical text, patient identifiers and
coordinates never leave the Health-flow process through this integration.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import JevProviderError
from app.harness.actions import HarnessAction, PlannedAction
from app.harness.planner import DeterministicPlanner, Planner
from app.harness.policy import HarnessPolicy
from app.harness.state import HarnessState

logger = logging.getLogger(__name__)

A = HarnessAction

_ACTION_DESCRIPTIONS: dict[HarnessAction, str] = {
    A.SAFETY_PRECHECK: "Run deterministic raw-text safety rules before any remote or LLM step.",
    A.EXTRACT_SYMPTOMS: "Extract structured symptoms with the local LLM.",
    A.LOAD_PATIENT_CONTEXT: "Load the minimum authorized patient context needed by later steps.",
    A.RUN_SAFETY_ASSESSMENT: "Run the complete deterministic safety assessment.",
    A.RUN_ML: "Run the auxiliary routing classifier after safety checks.",
    A.APPLY_ROUTING: "Combine safety floors and ML output into the routing decision.",
    A.SEARCH_FACILITIES: "Search compatible nearby facilities after routing is decided.",
    A.FINALIZE: "Finish the workflow when a valid result or controlled error is ready.",
}


class JevPlannerState(BaseModel):
    """Sanitized state allowed to leave the process for planner decisions."""

    model_config = ConfigDict(frozen=True)

    intent: str
    safety_precheck_complete: bool
    has_red_flag: bool
    symptoms_extracted: bool
    patient_context_loaded: bool
    safety_assessment_available: bool
    ml_prediction_available: bool
    routing_decision_available: bool
    facility_results_available: bool
    fatal_error_present: bool
    completed_actions: list[str]
    allowed_actions: list[str]
    step_count: int


class JevChoiceQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class JevRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    state: JevPlannerState
    questions: dict[str, JevChoiceQuestion]


class JevChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None


class JevUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = None
    output_tokens: int | None = None


class JevResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    answers: dict[str, JevChoiceAnswer]
    usage: JevUsage | None = None


@dataclass(frozen=True)
class JevDecision:
    action: HarnessAction
    probability: float
    model: str
    latency_ms: float


class JevDecisionProvider(Protocol):
    async def choose_next_action(
        self,
        *,
        state: HarnessState,
        allowed_actions: tuple[HarnessAction, ...],
    ) -> JevDecision: ...


class JevHttpProvider:
    """Small server-side HTTP client for POST /v1/systemone."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/v1/systemone"
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def choose_next_action(
        self,
        *,
        state: HarnessState,
        allowed_actions: tuple[HarnessAction, ...],
    ) -> JevDecision:
        if len(allowed_actions) < 2:
            raise JevProviderError("Jev choice requires at least two allowed actions")

        safe_state = _build_safe_state(state, allowed_actions)
        criteria = {action.value: _ACTION_DESCRIPTIONS[action] for action in allowed_actions}
        payload = JevRequest(
            model=self._model,
            state=safe_state,
            questions={
                "next_action": JevChoiceQuestion(
                    instructions=(
                        "Choose the safest and most appropriate next workflow action. "
                        "Use only one of the provided action labels. Safety and policy "
                        "constraints are authoritative."
                    ),
                    criteria=criteria,
                )
            },
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": _idempotency_key(state, allowed_actions),
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload.model_dump(mode="json"),
                )
            if response.status_code >= 400:
                raise JevProviderError(f"Jev HTTP status {response.status_code}")
            parsed = JevResponse.model_validate(response.json())
        except JevProviderError:
            raise
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise JevProviderError(f"invalid or unavailable Jev response: {type(exc).__name__}") from exc

        answer = parsed.answers.get("next_action")
        if answer is None:
            raise JevProviderError("Jev response did not include next_action")
        try:
            action = HarnessAction(answer.choice)
        except ValueError as exc:
            raise JevProviderError("Jev returned an unknown action") from exc
        probability = answer.probabilities.get(answer.choice)
        if probability is None:
            raise JevProviderError("Jev response omitted probability for the selected action")
        return JevDecision(
            action=action,
            probability=probability,
            model=parsed.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


class JevPlanner:
    """Experimental remote planner with deterministic, safety-preserving fallback."""

    def __init__(
        self,
        *,
        provider: JevDecisionProvider,
        policy: HarnessPolicy,
        min_probability: float,
        fallback: Planner | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._min_probability = min_probability
        self._fallback = fallback or DeterministicPlanner()

    async def next_action(self, state: HarnessState) -> PlannedAction:
        # Safety precheck and red-flag paths must never depend on a remote planner.
        if not state.done(A.SAFETY_PRECHECK) or state.has_red_flag or state.fatal_error is not None:
            return await self._fallback.next_action(state)

        allowed = self._policy.allowed_actions(state)
        if len(allowed) < 2:
            return await self._fallback.next_action(state)

        try:
            decision = await self._provider.choose_next_action(
                state=state,
                allowed_actions=allowed,
            )
        except JevProviderError as exc:
            logger.warning(
                "jev_planner_fallback",
                extra={
                    "request_id": state.request_id,
                    "step_count": state.step_count,
                    "error_type": type(exc).__name__,
                },
            )
            return await self._fallback.next_action(state)

        if decision.action not in allowed or decision.probability < self._min_probability:
            logger.info(
                "jev_planner_fallback",
                extra={
                    "request_id": state.request_id,
                    "step_count": state.step_count,
                    "selected_action": decision.action,
                    "probability": decision.probability,
                    "model": decision.model,
                    "fallback_reason": (
                        "action_not_allowed"
                        if decision.action not in allowed
                        else "probability_below_threshold"
                    ),
                },
            )
            return await self._fallback.next_action(state)

        planned = PlannedAction(
            action=decision.action,
            reason_code=f"JEV_PLANNER:{decision.action.value}",
        )
        # Validate here before returning so a bad remote proposal becomes a safe fallback.
        try:
            self._policy.validate(planned, state)
        except Exception as exc:
            # validate() only raises harness guard exceptions; never log exception text.
            logger.warning(
                "jev_planner_fallback",
                extra={
                    "request_id": state.request_id,
                    "step_count": state.step_count,
                    "error_type": type(exc).__name__,
                    "fallback_reason": "policy_rejected",
                },
            )
            return await self._fallback.next_action(state)

        logger.info(
            "jev_planner_decision",
            extra={
                "request_id": state.request_id,
                "step_count": state.step_count,
                "selected_action": decision.action,
                "probability": decision.probability,
                "model": decision.model,
                "latency_ms": decision.latency_ms,
            },
        )
        return planned


def _build_safe_state(
    state: HarnessState,
    allowed_actions: tuple[HarnessAction, ...],
) -> JevPlannerState:
    return JevPlannerState(
        intent=state.intent.value,
        safety_precheck_complete=state.done(A.SAFETY_PRECHECK),
        has_red_flag=state.has_red_flag,
        symptoms_extracted=state.extracted_symptoms is not None,
        patient_context_loaded=state.patient_context is not None,
        safety_assessment_available=state.safety_assessment is not None,
        ml_prediction_available=state.ml_prediction is not None,
        routing_decision_available=state.routing_decision is not None,
        facility_results_available=bool(state.facilities),
        fatal_error_present=state.fatal_error is not None,
        completed_actions=[action.value for action in state.completed_actions],
        allowed_actions=[action.value for action in allowed_actions],
        step_count=state.step_count,
    )


def _idempotency_key(
    state: HarnessState,
    allowed_actions: tuple[HarnessAction, ...],
) -> str:
    raw = f"{state.request_id}:{state.step_count}:{','.join(a.value for a in allowed_actions)}"
    return hashlib.sha256(raw.encode()).hexdigest()
