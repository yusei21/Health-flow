"""Planners choose the next action from `HarnessState`.

A planner only PROPOSES; `HarnessPolicy` validates every proposal before execution, so
a buggy or future LLM-backed planner cannot skip safety, call the LLM after a precheck
red flag, run ML on a red flag, or exceed budgets.
"""

from typing import Protocol

from app.harness.actions import HarnessAction, PlannedAction, ReasonCode
from app.harness.state import HarnessState

A = HarnessAction
R = ReasonCode


class Planner(Protocol):
    def next_action(self, state: HarnessState) -> PlannedAction: ...


class DeterministicPlanner:
    """Rule-based planner for CARE_ROUTING; skips steps the state makes unnecessary.

    Non-critical: PRECHECK → EXTRACT → CONTEXT → SAFETY → ML → ROUTING → FACILITIES → FINALIZE
    Precheck red flag: PRECHECK → ROUTING → FACILITIES → FINALIZE (no LLM, no ML)
    Red flag found after extraction: ... → SAFETY → ROUTING → FACILITIES → FINALIZE (no ML)
    LLM failure without red flag: PRECHECK → EXTRACT → FINALIZE (controlled error)
    """

    def next_action(self, state: HarnessState) -> PlannedAction:
        if not state.done(A.SAFETY_PRECHECK):
            return PlannedAction(A.SAFETY_PRECHECK, R.START_WITH_SAFETY_PRECHECK)
        precheck = state.safety_precheck
        if precheck is not None and precheck.has_red_flag:
            return self._route_and_finish(state, R.PRECHECK_RED_FLAG_SKIP_LLM_AND_ML)

        if not state.done(A.EXTRACT_SYMPTOMS):
            return PlannedAction(A.EXTRACT_SYMPTOMS, R.NO_PRECHECK_RED_FLAG)
        if state.extracted_symptoms is None:
            # Text rules already ran in the precheck without a red flag; without an
            # extraction no structured rule can fire, so there is nothing safe to route.
            return PlannedAction(A.FINALIZE, R.LLM_FAILED_NO_RED_FLAG)
        if not state.done(A.LOAD_PATIENT_CONTEXT):
            return PlannedAction(A.LOAD_PATIENT_CONTEXT, R.SYMPTOMS_EXTRACTED)
        if not state.done(A.RUN_SAFETY_ASSESSMENT):
            return PlannedAction(A.RUN_SAFETY_ASSESSMENT, R.CONTEXT_LOADED)
        if state.has_red_flag:
            return self._route_and_finish(state, R.RED_FLAG_SKIP_ML)
        if not state.done(A.RUN_ML):
            return PlannedAction(A.RUN_ML, R.NO_RED_FLAG_RUN_ML)
        return self._route_and_finish(state, R.ML_STEP_DONE)

    @staticmethod
    def _route_and_finish(state: HarnessState, routing_reason: ReasonCode) -> PlannedAction:
        if state.routing_decision is None:
            return PlannedAction(A.APPLY_ROUTING, routing_reason)
        if not state.done(A.SEARCH_FACILITIES):
            return PlannedAction(A.SEARCH_FACILITIES, R.ROUTING_DECIDED)
        return PlannedAction(A.FINALIZE, R.FACILITY_SEARCH_DONE)
