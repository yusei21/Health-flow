from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.harness.actions import FinishReason, HarnessAction, Intent
from app.safety.schemas import SafetyAssessment
from app.schemas.care import most_severe
from app.schemas.facility import FacilityMatch
from app.schemas.patient import PatientContext
from app.schemas.routing import MLPrediction, RoutingDecision
from app.schemas.symptoms import SymptomExtraction


class StageError(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: HarnessAction
    error_type: str


class ActionRecord(BaseModel):
    """One executed step. Audit data only: no clinical text, no record content."""

    model_config = ConfigDict(frozen=True)

    step: int
    action: HarnessAction
    reason_code: str
    status: str  # "ok" | "handled_error:<Type>" | "error:<Type>"
    duration_ms: float


class HarnessState(BaseModel):
    """Everything known about one routing request, filled action by action.

    `user_message` is kept in memory only; it is never logged nor persisted here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    user_id: UUID
    user_message: str = Field(repr=False)
    latitude: float
    longitude: float
    intent: Intent = Intent.CARE_ROUTING
    use_patient_record: bool = False

    safety_precheck: SafetyAssessment | None = None  # raw text only, before any LLM call
    extracted_symptoms: SymptomExtraction | None = None
    patient_context: PatientContext | None = None
    safety_assessment: SafetyAssessment | None = None  # text + extraction + context
    ml_prediction: MLPrediction | None = None
    routing_decision: RoutingDecision | None = None
    facilities: list[FacilityMatch] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)

    completed_actions: list[HarnessAction] = Field(default_factory=list)
    action_history: list[ActionRecord] = Field(default_factory=list)
    step_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    finished: bool = False
    finish_reason: FinishReason | None = None
    # Error to surface to the caller when the run cannot produce a decision.
    fatal_error: Exception | None = Field(default=None, exclude=True, repr=False)

    def done(self, action: HarnessAction) -> bool:
        return action in self.completed_actions

    @property
    def effective_safety(self) -> SafetyAssessment | None:
        """Most protective combination of the precheck and the full assessment.

        A later assessment can only add rules or raise the floor, never remove them.
        """
        return combine_assessments(self.safety_precheck, self.safety_assessment)

    @property
    def has_red_flag(self) -> bool:
        safety = self.effective_safety
        return safety is not None and safety.has_red_flag


def combine_assessments(*assessments: SafetyAssessment | None) -> SafetyAssessment | None:
    present = [a for a in assessments if a is not None]
    if not present:
        return None
    rules = list(dict.fromkeys(rule for a in present for rule in a.matched_rules))
    return SafetyAssessment(
        has_red_flag=any(a.has_red_flag for a in present),
        matched_rules=rules,
        suggested_minimum_care_level=most_severe(
            *(a.suggested_minimum_care_level for a in present)
        ),
    )
