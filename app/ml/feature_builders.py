"""Feature sets: canonical training example → numeric vector.

One builder per feature set; no `if dataset == ...` branching. `app.ml.features` stays
the single source of truth for the Health-flow symptom features used by the API.
Missing values are NaN and are imputed inside the sklearn Pipeline, never here.
"""

import math
from typing import Protocol

from app.ml.data.schemas import RoutingTrainingExample
from app.ml.features import FEATURE_NAMES, build_features
from app.schemas.patient import AgeRange, PatientContext
from app.schemas.symptoms import Severity, SymptomExtraction


class FeatureBuilder(Protocol):
    feature_set: str
    feature_names: tuple[str, ...]

    def build(self, example: RoutingTrainingExample) -> list[float]: ...


class HealthFlowSymptomFeatureBuilder:
    """Same features the API computes at inference time (symptoms + minimal context)."""

    feature_set: str = "healthflow-symptoms-v1"
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def build(self, example: RoutingTrainingExample) -> list[float]:
        extraction = SymptomExtraction(
            symptoms=example.symptoms or [],
            severity=example.severity or Severity.UNKNOWN,
            duration_minutes=example.duration_minutes,
        )
        context = PatientContext(
            age_range=example.age_range or AgeRange.UNKNOWN,
            risk_factors=frozenset(example.risk_factors or ()),
        )
        return build_features(extraction, context)


class MimicStructuredFeatureBuilder:
    """Triage vital signs + pain only. Age is absent from the triage table, so unused."""

    feature_set: str = "mimic-structured-vitals-v1"
    feature_names: tuple[str, ...] = (
        "heart_rate",
        "respiratory_rate",
        "oxygen_saturation",
        "systolic_bp",
        "diastolic_bp",
        "temperature_celsius",
        "pain",
    )

    def build(self, example: RoutingTrainingExample) -> list[float]:
        values: list[float | None] = [getattr(example, name) for name in self.feature_names]
        return [math.nan if value is None else value for value in values]


FEATURE_BUILDERS: dict[str, FeatureBuilder] = {
    builder.feature_set: builder
    for builder in (HealthFlowSymptomFeatureBuilder(), MimicStructuredFeatureBuilder())
}
