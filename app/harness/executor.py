"""Runs one validated action against the matching component.

Handlers live in a typed registry (one per `HarnessAction`); there is no free-form
dispatch. Failure handling is unchanged from the linear harness: LLM failure is recorded
and surfaced only when no safe decision exists, ML failure falls back conservatively in
the routing agent, and facility failure keeps the routing decision.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.intent_agent import IntentAgent
from app.agents.navigation_agent import NavigationAgent
from app.agents.patient_context_agent import PatientContextAgent
from app.core.exceptions import FacilityProviderError, LLMError, MLInferenceError
from app.core.logging import trace_stage
from app.harness.actions import ACTION_KIND, ActionKind, FinishReason, HarnessAction, PlannedAction
from app.harness.state import ActionRecord, HarnessState, StageError
from app.ml.inference import RoutingInferenceService
from app.safety.engine import SafetyEngine

logger = logging.getLogger(__name__)

A = HarnessAction
Handler = Callable[[HarnessState], Awaitable[None]]


class ActionExecutor:
    def __init__(
        self,
        intent_agent: IntentAgent,
        context_agent: PatientContextAgent,
        safety_engine: SafetyEngine,
        inference: RoutingInferenceService,
        routing_agent: CareRoutingAgent,
        navigation_agent: NavigationAgent,
    ) -> None:
        self._intent = intent_agent
        self._context = context_agent
        self._safety = safety_engine
        self._inference = inference
        self._routing = routing_agent
        self._navigation = navigation_agent
        self._handlers: dict[HarnessAction, Handler] = {
            A.SAFETY_PRECHECK: self._safety_precheck,
            A.EXTRACT_SYMPTOMS: self._extract_symptoms,
            A.LOAD_PATIENT_CONTEXT: self._load_patient_context,
            A.RUN_SAFETY_ASSESSMENT: self._run_safety_assessment,
            A.RUN_ML: self._run_ml,
            A.APPLY_ROUTING: self._apply_routing,
            A.SEARCH_FACILITIES: self._search_facilities,
            A.FINALIZE: self._finalize,
        }
        if missing := set(HarnessAction) - set(self._handlers):
            raise ValueError(f"no handler for actions: {sorted(missing)}")

    async def execute(self, planned: PlannedAction, state: HarnessState) -> None:
        action = planned.action
        state.step_count += 1
        kind = ACTION_KIND[action]
        # Budgets count attempts, including failed ones.
        if kind is ActionKind.LLM:
            state.llm_call_count += 1
        elif kind is ActionKind.TOOL:
            state.tool_call_count += 1
        errors_before = len(state.errors)
        status = "ok"
        started = time.perf_counter()
        try:
            with trace_stage(logger, action):
                await self._handlers[action](state)
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            raise
        finally:
            if len(state.errors) > errors_before:
                status = f"handled_error:{state.errors[-1].error_type}"
            state.completed_actions.append(action)
            state.action_history.append(
                ActionRecord(
                    step=state.step_count,
                    action=action,
                    reason_code=planned.reason_code,
                    status=status,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            )

    async def _safety_precheck(self, state: HarnessState) -> None:
        # Deterministic text rules on the raw report, before any LLM call.
        state.safety_precheck = self._safety.assess(state.user_message, None, None)
        logger.info(
            "safety_precheck_result",
            extra={
                "red_flag": state.safety_precheck.has_red_flag,
                "matched_rules": state.safety_precheck.matched_rules,
            },
        )

    async def _extract_symptoms(self, state: HarnessState) -> None:
        try:
            state.extracted_symptoms = await self._intent.extract(state.user_message)
        except LLMError as exc:
            state.errors.append(StageError(stage=A.EXTRACT_SYMPTOMS, error_type=type(exc).__name__))
            state.fatal_error = exc

    async def _load_patient_context(self, state: HarnessState) -> None:
        state.patient_context = await self._context.build_context(
            state.user_id, state.extracted_symptoms
        )

    async def _run_safety_assessment(self, state: HarnessState) -> None:
        state.safety_assessment = self._safety.assess(
            state.user_message, state.extracted_symptoms, state.patient_context
        )
        logger.info(
            "safety_result",
            extra={
                "red_flag": state.safety_assessment.has_red_flag,
                "matched_rules": state.safety_assessment.matched_rules,
            },
        )

    async def _run_ml(self, state: HarnessState) -> None:
        assert state.extracted_symptoms is not None  # noqa: S101 - enforced by policy
        assert state.patient_context is not None  # noqa: S101 - enforced by policy
        try:
            state.ml_prediction = self._inference.predict(
                state.extracted_symptoms, state.patient_context
            )
        except MLInferenceError as exc:
            # Degrade safely: the routing agent applies a conservative fallback.
            state.errors.append(StageError(stage=A.RUN_ML, error_type=type(exc).__name__))

    async def _apply_routing(self, state: HarnessState) -> None:
        safety = state.effective_safety
        assert safety is not None  # noqa: S101 - enforced by policy
        state.routing_decision = self._routing.decide(safety, state.ml_prediction)
        logger.info(
            "routing_result",
            extra={
                "care_level": state.routing_decision.care_level,
                "safety_override": state.routing_decision.safety_override,
            },
        )

    async def _search_facilities(self, state: HarnessState) -> None:
        assert state.routing_decision is not None  # noqa: S101 - enforced by policy
        try:
            state.facilities = await self._navigation.find_facilities(
                state.routing_decision.service_type, state.latitude, state.longitude
            )
        except FacilityProviderError as exc:
            # The care level is still valid guidance even without a facility.
            state.errors.append(
                StageError(stage=A.SEARCH_FACILITIES, error_type=type(exc).__name__)
            )

    async def _finalize(self, state: HarnessState) -> None:
        state.finished = True
        state.finish_reason = (
            FinishReason.COMPLETED
            if state.routing_decision is not None
            else FinishReason.LLM_UNAVAILABLE
        )
