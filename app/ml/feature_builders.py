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


# Structured triage vitals shared by MIMIC-IV-ED and Triagegeist (canonical field names).
STRUCTURED_VITAL_FEATURES: tuple[str, ...] = (
    "heart_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "systolic_bp",
    "diastolic_bp",
    "temperature_celsius",
    "pain",
)
_AGE_RANGES_BY_RANK = [r for r in AgeRange if r is not AgeRange.UNKNOWN]


def _structured_vitals(example: RoutingTrainingExample) -> list[float]:
    values: list[float | None] = [getattr(example, name) for name in STRUCTURED_VITAL_FEATURES]
    return [math.nan if value is None else value for value in values]


class MimicStructuredFeatureBuilder:
    """Triage vital signs + pain only. Age is absent from the triage table, so unused."""

    feature_set: str = "mimic-structured-vitals-v1"
    feature_names: tuple[str, ...] = STRUCTURED_VITAL_FEATURES

    def build(self, example: RoutingTrainingExample) -> list[float]:
        return _structured_vitals(example)


class TriagegeistStructuredFeatureBuilder:
    """Triage vitals + pain + ordinal age range (NaN when unknown). No IDs, no free text.

    Separate feature set from MIMIC so either can evolve without invalidating the other's
    models. Fields absent from the real CSV are NaN in every row (see dataset metadata).
    """

    feature_set: str = "triagegeist-structured-v1"
    feature_names: tuple[str, ...] = (*STRUCTURED_VITAL_FEATURES, "age_range_rank")

    def build(self, example: RoutingTrainingExample) -> list[float]:
        age = example.age_range
        age_rank = (
            math.nan
            if age is None or age is AgeRange.UNKNOWN
            else float(_AGE_RANGES_BY_RANK.index(age))
        )
        return [*_structured_vitals(example), age_rank]


FEATURE_BUILDERS: dict[str, FeatureBuilder] = {
    builder.feature_set: builder
    for builder in (
        HealthFlowSymptomFeatureBuilder(),
        MimicStructuredFeatureBuilder(),
        TriagegeistStructuredFeatureBuilder(),
    )
}
