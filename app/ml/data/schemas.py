"""Canonical training schema shared by every dataset source.

Each source (synthetic, MIMIC-IV-ED, ...) converts its rows into
`RoutingTrainingExample`. Feature builders read only this schema, so new sources or
features do not require rewriting loaders. `None` always means "not available".
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange, RiskFactor
from app.schemas.symptoms import Severity, Symptom


class RoutingTrainingExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Pseudonymous patient key used ONLY to keep a patient on one side of the split.
    # Never a feature, never logged.
    group_id: str | None = None
    label: CareLevel

    symptoms: list[Symptom] | None = None
    severity: Severity | None = None
    duration_minutes: int | None = Field(default=None, ge=0)
    age_range: AgeRange | None = None
    risk_factors: list[RiskFactor] | None = None

    heart_rate: float | None = None
    respiratory_rate: float | None = None
    oxygen_saturation: float | None = None
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    temperature_celsius: float | None = None
    pain: float | None = None

    # Free text kept for the future LLM experiment (C); never a direct feature.
    chief_complaint: str | None = None


class DatasetInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    disclaimer: str
    label_definition: str


@dataclass(frozen=True)
class TrainingDataset:
    info: DatasetInfo
    examples: list[RoutingTrainingExample]

    @property
    def number_of_patients(self) -> int | None:
        """Distinct patients, or None when the source has no patient identity."""
        groups = {ex.group_id for ex in self.examples}
        return None if None in groups else len(groups)
