from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange
from app.schemas.symptoms import Severity, Symptom

ACADEMIC_SOURCE = "ACADEMIC_SIMULATED — not an official SUS protocol"


class SafetyRule(BaseModel):
    """Declarative rule so rules can later be loaded from validated sources.

    A rule matches when its structured condition holds (all of `all_symptoms`, at least
    one of `any_symptoms`, the severity floor and the age ranges), OR when any of
    `text_patterns` appears in the normalized report. Text patterns are a fail-safe
    for when the LLM misses or cannot extract a critical sign.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Z_]+_\d{3}$")
    description: str
    minimum_care_level: CareLevel
    all_symptoms: frozenset[Symptom] = frozenset()
    any_symptoms: frozenset[Symptom] = frozenset()
    min_severity: Severity | None = None
    age_ranges: frozenset[AgeRange] = frozenset()
    text_patterns: tuple[str, ...] = ()
    source: str = ACADEMIC_SOURCE

    @model_validator(mode="after")
    def _require_a_trigger(self) -> Self:
        if not (self.all_symptoms or self.any_symptoms or self.text_patterns):
            raise ValueError(f"rule {self.id} has no trigger")
        return self


class SafetyAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    has_red_flag: bool
    matched_rules: list[str]
    suggested_minimum_care_level: CareLevel | None
