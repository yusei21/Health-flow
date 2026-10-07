from datetime import date
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AgeRange(StrEnum):
    UNKNOWN = "unknown"
    INFANT = "infant"  # < 2
    CHILD = "child"  # 2-11
    ADOLESCENT = "adolescent"  # 12-17
    ADULT = "adult"  # 18-59
    ELDERLY = "elderly"  # 60+

    @classmethod
    def from_age(cls, age: int | None) -> "AgeRange":
        if age is None:
            return cls.UNKNOWN
        if age < 2:
            return cls.INFANT
        if age < 12:
            return cls.CHILD
        if age < 18:
            return cls.ADOLESCENT
        if age < 60:
            return cls.ADULT
        return cls.ELDERLY


class RiskFactor(StrEnum):
    """Coarse condition categories; the only clinical history the ML model sees."""

    CARDIOVASCULAR_DISEASE = "cardiovascular_disease"
    CHRONIC_RESPIRATORY_DISEASE = "chronic_respiratory_disease"
    DIABETES = "diabetes"
    IMMUNOSUPPRESSION = "immunosuppression"
    PREGNANCY = "pregnancy"


class Condition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(max_length=200)
    risk_factor: RiskFactor | None = None


class Medication(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(max_length=200)
    is_anticoagulant: bool = False


class Encounter(BaseModel):
    model_config = ConfigDict(frozen=True)

    occurred_on: date
    service: str = Field(max_length=100)
    summary: str = Field(max_length=500)


class PatientRecord(BaseModel):
    """Simplified academic record. Real integrations require formal authorization."""

    model_config = ConfigDict(frozen=True)

    patient_id: UUID
    user_id: UUID
    display_name: str = Field(max_length=200)
    age: int | None = Field(default=None, ge=0, le=130)
    allergies: list[str] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    active_medications: list[Medication] = Field(default_factory=list)
    previous_encounters: list[Encounter] = Field(default_factory=list)


class PatientContext(BaseModel):
    """Minimum record subset relevant to the current report (data minimization)."""

    model_config = ConfigDict(frozen=True)

    age_range: AgeRange = AgeRange.UNKNOWN
    risk_factors: frozenset[RiskFactor] = frozenset()
    relevant_conditions: list[str] = Field(default_factory=list)
    active_medications: list[str] = Field(default_factory=list)
    relevant_allergies: list[str] = Field(default_factory=list)
