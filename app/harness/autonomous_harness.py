"""State-driven Agent Harness: plan → validate → execute, until FINALIZE or a guard.

    while not state.finished:
        planned = planner.next_action(state)     # proposes one HarnessAction
        policy.validate(planned, state)          # safety order, allowed set, budgets
        await executor.execute(planned, state)   # registry handler, records history

The LLM only interprets, the ML only suggests, the SafetyEngine sets floors that nothing
may lower, and the policy (not the planner) has the final word on what may run.
"""

import asyncio
import logging
from uuid import UUID

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.intent_agent import IntentAgent
from app.agents.navigation_agent import NavigationAgent
from app.agents.patient_context_agent import PatientContextAgent
from app.core.exceptions import HarnessError, HarnessTimeoutError
from app.harness.actions import FinishReason
from app.harness.executor import ActionExecutor
from app.harness.planner import DeterministicPlanner, Planner
from app.harness.policy import HarnessPolicy
from app.harness.state import HarnessState
from app.ml.inference import RoutingInferenceService
from app.safety.engine import SafetyEngine

logger = logging.getLogger(__name__)


class AutonomousHealthFlowHarness:
    def __init__(
        self,
        intent_agent: IntentAgent,
        context_agent: PatientContextAgent,
        safety_engine: SafetyEngine,
        inference: RoutingInferenceService,
        routing_agent: CareRoutingAgent,
        navigation_agent: NavigationAgent,
        planner: Planner | None = None,
        policy: HarnessPolicy | None = None,
    ) -> None:
        self._executor = ActionExecutor(
            intent_agent, context_agent, safety_engine, inference, routing_agent, navigation_agent
        )
        self._planner = planner or DeterministicPlanner()
        self._policy = policy or HarnessPolicy()

    async def run(
        self, request_id: str, user_id: UUID, message: str, latitude: float, longitude: float
    ) -> HarnessState:
        state = HarnessState(
            request_id=request_id,
            user_id=user_id,
            user_message=message,
            latitude=latitude,
            longitude=longitude,
        )
        deadline = asyncio.timeout(self._policy.limits.timeout_seconds)
        try:
            async with deadline:
                await self._loop(state)
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            self._abort(state, FinishReason.TIMEOUT)
            raise HarnessTimeoutError("global timeout reached", FinishReason.TIMEOUT) from exc
        except HarnessError as exc:
            self._abort(state, FinishReason(exc.finish_reason))
            raise

        if state.finish_reason is FinishReason.LLM_UNAVAILABLE:
            # No extraction and no red flag: surface the controlled LLM error.
            assert state.fatal_error is not None  # noqa: S101 - set with this finish reason
            raise state.fatal_error
        return state

    async def _loop(self, state: HarnessState) -> None:
        while not state.finished:
            planned = await self._planner.next_action(state)
            # Audit trail: ids, action names and reason codes only — never clinical text.
            logger.info(
                "planner_decision",
                extra={
                    "request_id": state.request_id,
                    "current_action": state.completed_actions[-1]
                    if state.completed_actions
                    else None,
                    "next_action": planned.action,
                    "reason_code": planned.reason_code,
                    "step_count": state.step_count,
                },
            )
            self._policy.validate(planned, state)
            await self._executor.execute(planned, state)
        logger.info("harness_finished", extra=_summary(state))

    @staticmethod
    def _abort(state: HarnessState, reason: FinishReason) -> None:
        state.finished = True
        state.finish_reason = reason
        logger.warning("harness_aborted", extra=_summary(state))


def _summary(state: HarnessState) -> dict[str, object]:
    return {
        "request_id": state.request_id,
        "finish_reason": state.finish_reason,
        "step_count": state.step_count,
        "llm_call_count": state.llm_call_count,
        "tool_call_count": state.tool_call_count,
        "actions": [str(action) for action in state.completed_actions],
    }
