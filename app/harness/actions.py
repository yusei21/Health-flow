"""Closed vocabulary of what the harness may do.

Autonomy here means choosing the NEXT step among these actions; nothing outside this
enum can be executed, whoever the planner is (deterministic today, LLM in the future).
"""

from dataclasses import dataclass
from enum import StrEnum


class HarnessAction(StrEnum):
    SAFETY_PRECHECK = "safety_precheck"
    EXTRACT_SYMPTOMS = "extract_symptoms"
    LOAD_PATIENT_CONTEXT = "load_patient_context"
    RUN_SAFETY_ASSESSMENT = "run_safety_assessment"
    RUN_ML = "run_ml"
    APPLY_ROUTING = "apply_routing"
    SEARCH_FACILITIES = "search_facilities"
    FINALIZE = "finalize"


class ActionKind(StrEnum):
    LLM = "llm"  # counts against max_llm_calls
    TOOL = "tool"  # repository / model / facility provider; counts against max_tool_calls
    INTERNAL = "internal"  # deterministic, in-process


ACTION_KIND: dict[HarnessAction, ActionKind] = {
    HarnessAction.SAFETY_PRECHECK: ActionKind.INTERNAL,
    HarnessAction.EXTRACT_SYMPTOMS: ActionKind.LLM,
    HarnessAction.LOAD_PATIENT_CONTEXT: ActionKind.TOOL,
    HarnessAction.RUN_SAFETY_ASSESSMENT: ActionKind.INTERNAL,
    HarnessAction.RUN_ML: ActionKind.TOOL,
    HarnessAction.APPLY_ROUTING: ActionKind.INTERNAL,
    HarnessAction.SEARCH_FACILITIES: ActionKind.TOOL,
    HarnessAction.FINALIZE: ActionKind.INTERNAL,
}


class Intent(StrEnum):
    CARE_ROUTING = "care_routing"
    MEDICATION = "medication"  # planned (Etapa 2): no actions/agent yet
    INSURANCE = "insurance"  # planned (Etapa 3): no actions/agent yet


# An intent with no entry here cannot execute anything (the policy rejects it).
ALLOWED_ACTIONS: dict[Intent, frozenset[HarnessAction]] = {
    Intent.CARE_ROUTING: frozenset(HarnessAction),
}


class ReasonCode(StrEnum):
    """Why the planner chose an action. Auditable; never contains clinical text."""

    START_WITH_SAFETY_PRECHECK = "START_WITH_SAFETY_PRECHECK"
    PRECHECK_RED_FLAG_SKIP_LLM_AND_ML = "PRECHECK_RED_FLAG_SKIP_LLM_AND_ML"
    NO_PRECHECK_RED_FLAG = "NO_PRECHECK_RED_FLAG"
    SYMPTOMS_EXTRACTED = "SYMPTOMS_EXTRACTED"
    CONTEXT_LOADED = "CONTEXT_LOADED"
    RED_FLAG_SKIP_ML = "RED_FLAG_SKIP_ML"
    NO_RED_FLAG_RUN_ML = "NO_RED_FLAG_RUN_ML"
    ML_STEP_DONE = "ML_STEP_DONE"
    ROUTING_DECIDED = "ROUTING_DECIDED"
    FACILITY_SEARCH_DONE = "FACILITY_SEARCH_DONE"
    LLM_FAILED_NO_RED_FLAG = "LLM_FAILED_NO_RED_FLAG"


@dataclass(frozen=True)
class PlannedAction:
    action: HarnessAction
    reason_code: str


class FinishReason(StrEnum):
    COMPLETED = "completed"
    LLM_UNAVAILABLE = "llm_unavailable"  # no extraction and no red flag: controlled error
    POLICY_VIOLATION = "policy_violation"
    STEP_LIMIT = "step_limit"
    LLM_CALL_LIMIT = "llm_call_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TIMEOUT = "timeout"
