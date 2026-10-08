"""Policy/guard layer: every planned action is validated here before it runs.

Safety has authority over the planner. These checks hold for ANY planner, so a remote
planner cannot reorder or skip the safety-critical parts of the workflow.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

from app.core.exceptions import HarnessError, HarnessLimitError, HarnessPolicyError
from app.harness.actions import (
    ACTION_KIND,
    ALLOWED_ACTIONS,
    ActionKind,
    FinishReason,
    HarnessAction,
    PlannedAction,
)
from app.harness.state import HarnessState

A = HarnessAction


@dataclass(frozen=True)
class HarnessLimits:
    max_steps: int = 10
    max_llm_calls: int = 1
    max_tool_calls: int = 3
    timeout_seconds: float = 90.0


def _precheck_required(state: HarnessState) -> str | None:
    return None if state.done(A.SAFETY_PRECHECK) else "safety precheck must run first"


def _extract_symptoms(state: HarnessState) -> str | None:
    if reason := _precheck_required(state):
        return reason
    precheck = state.safety_precheck
    if precheck is not None and precheck.has_red_flag:
        return "precheck red flag: routing must not wait for the LLM"
    return None


def _load_patient_context(state: HarnessState) -> str | None:
    if reason := _precheck_required(state):
        return reason
    if not state.done(A.EXTRACT_SYMPTOMS) or state.extracted_symptoms is None:
        return "patient context requires a successful symptom extraction"
    return None


def _run_safety_assessment(state: HarnessState) -> str | None:
    if reason := _precheck_required(state):
        return reason
    if state.extracted_symptoms is None:
        return "full safety assessment requires extracted symptoms"
    if state.patient_context is None:
        return "full safety assessment requires patient context"
    return None


def _run_ml(state: HarnessState) -> str | None:
    if not state.done(A.RUN_SAFETY_ASSESSMENT):
        return "ML may only run after the full safety assessment"
    if state.has_red_flag:
        return "red flag: ML must not run"
    if state.extracted_symptoms is None or state.patient_context is None:
        return "ML needs extracted symptoms and patient context"
    return None


def _apply_routing(state: HarnessState) -> str | None:
    if reason := _precheck_required(state):
        return reason
    precheck = state.safety_precheck
    precheck_red_flag = precheck is not None and precheck.has_red_flag
    if not precheck_red_flag and not state.done(A.RUN_SAFETY_ASSESSMENT):
        return "routing needs the full safety assessment unless the precheck found a red flag"
    if not state.has_red_flag and not state.done(A.RUN_ML):
        return "without a red flag the ML step must be attempted before routing"
    return None


def _search_facilities(state: HarnessState) -> str | None:
    return None if state.routing_decision is not None else "no routing decision yet"


def _finalize(state: HarnessState) -> str | None:
    if state.routing_decision is None and state.fatal_error is None:
        return "nothing to finalize: no routing decision and no controlled error"
    return None


_PRECONDITIONS: dict[HarnessAction, Callable[[HarnessState], str | None]] = {
    A.SAFETY_PRECHECK: lambda _: None,
    A.EXTRACT_SYMPTOMS: _extract_symptoms,
    A.LOAD_PATIENT_CONTEXT: _load_patient_context,
    A.RUN_SAFETY_ASSESSMENT: _run_safety_assessment,
    A.RUN_ML: _run_ml,
    A.APPLY_ROUTING: _apply_routing,
    A.SEARCH_FACILITIES: _search_facilities,
    A.FINALIZE: _finalize,
}


class HarnessPolicy:
    def __init__(self, limits: HarnessLimits | None = None) -> None:
        self.limits = limits or HarnessLimits()

    def validate(self, planned: PlannedAction, state: HarnessState) -> None:
        action = planned.action
        if state.finished:
            self._reject(f"run already finished; refused {action}")
        if action not in ALLOWED_ACTIONS.get(state.intent, frozenset()):
            self._reject(f"action {action} not allowed for intent {state.intent}")
        if state.done(action):
            self._reject(f"action {action} already executed")
        self._check_budgets(action, state)
        if reason := _PRECONDITIONS[action](state):
            self._reject(f"{action} refused: {reason}")

    def allowed_actions(self, state: HarnessState) -> tuple[HarnessAction, ...]:
        """Return actions the policy would accept now, without mutating state."""
        allowed: list[HarnessAction] = []
        for action in HarnessAction:
            try:
                self.validate(PlannedAction(action, "POLICY_PROBE"), state)
            except HarnessError:
                continue
            allowed.append(action)
        return tuple(allowed)

    def _check_budgets(self, action: HarnessAction, state: HarnessState) -> None:
        limits = self.limits
        if state.step_count >= limits.max_steps:
            raise HarnessLimitError(
                f"step limit {limits.max_steps} reached", FinishReason.STEP_LIMIT
            )
        kind = ACTION_KIND[action]
        if kind is ActionKind.LLM and state.llm_call_count >= limits.max_llm_calls:
            raise HarnessLimitError(
                f"LLM call limit {limits.max_llm_calls} reached", FinishReason.LLM_CALL_LIMIT
            )
        if kind is ActionKind.TOOL and state.tool_call_count >= limits.max_tool_calls:
            raise HarnessLimitError(
                f"tool call limit {limits.max_tool_calls} reached", FinishReason.TOOL_CALL_LIMIT
            )

    @staticmethod
    def _reject(message: str) -> NoReturn:
        raise HarnessPolicyError(message, FinishReason.POLICY_VIOLATION)
