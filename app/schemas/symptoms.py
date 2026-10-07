from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Symptom(StrEnum):
    """Canonical symptom vocabulary shared by the LLM schema, rules and ML features."""

    CHEST_PAIN = "chest_pain"
    SHORTNESS_OF_BREATH = "shortness_of_breath"
    LOSS_OF_CONSCIOUSNESS = "loss_of_consciousness"
    SEIZURE = "seizure"
    FACIAL_DROOP = "facial_droop"
    SLURRED_SPEECH = "slurred_speech"
    ONE_SIDED_WEAKNESS = "one_sided_weakness"
    SEVERE_BLEEDING = "severe_bleeding"
    SELF_HARM_IDEATION = "self_harm_ideation"
    FEVER = "fever"
    COUGH = "cough"
    SORE_THROAT = "sore_throat"
    RUNNY_NOSE = "runny_nose"
    HEADACHE = "headache"
    ABDOMINAL_PAIN = "abdominal_pain"
    VOMITING = "vomiting"
    DIARRHEA = "diarrhea"
    DIZZINESS = "dizziness"
    RASH = "rash"
    BACK_PAIN = "back_pain"
    FATIGUE = "fatigue"
    INJURY = "injury"


class Severity(StrEnum):
    UNKNOWN = "unknown"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)


class SymptomExtraction(BaseModel):
    """Structured view of a free-text report. Describes symptoms; never a diagnosis."""

    model_config = ConfigDict(frozen=True)

    symptoms: list[Symptom] = Field(default_factory=list, max_length=len(Symptom))
    duration_minutes: int | None = Field(default=None, ge=0, le=60 * 24 * 365)
    severity: Severity = Severity.UNKNOWN
    age: int | None = Field(default=None, ge=0, le=130)

    @field_validator("symptoms")
    @classmethod
    def _deduplicate(cls, value: list[Symptom]) -> list[Symptom]:
        return list(dict.fromkeys(value))
