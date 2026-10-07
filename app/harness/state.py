from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.safety.schemas import SafetyAssessment
from app.schemas.facility import FacilityMatch
from app.schemas.patient import PatientContext
from app.schemas.routing import MLPrediction, RoutingDecision
from app.schemas.symptoms import SymptomExtraction


class Stage(StrEnum):
    EXTRACT_USER_REPORT = "extract_user_report"
    BUILD_PATIENT_CONTEXT = "build_patient_context"
    SAFETY_ASSESSMENT = "safety_assessment"
    ML_CLASSIFIER = "ml_classifier"
    ROUTING_RULES = "routing_rules"
    FIND_FACILITIES = "find_facilities"


class StageError(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Stage
    error_type: str


class HarnessState(BaseModel):
    """Everything known about one routing request, filled stage by stage.

    `user_message` is kept in memory only; it is never logged nor persisted here.
    """

    request_id: str
    user_id: UUID
    user_message: str = Field(repr=False)
    latitude: float
    longitude: float
    extracted_symptoms: SymptomExtraction | None = None
    patient_context: PatientContext | None = None
    safety_assessment: SafetyAssessment | None = None
    ml_prediction: MLPrediction | None = None
    routing_decision: RoutingDecision | None = None
    facilities: list[FacilityMatch] = Field(default_factory=list)
    errors: list[StageError] = Field(default_factory=list)
