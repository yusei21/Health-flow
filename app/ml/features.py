"""Feature engineering shared by training and inference (single source of truth).

Changing this module changes FEATURE_NAMES; inference refuses artifacts trained with a
different feature list, which prevents silent train/serve skew.
"""

import math

from app.schemas.patient import AgeRange, PatientContext, RiskFactor
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction

_AGE_RANGE_ORDER = [
    AgeRange.INFANT,
    AgeRange.CHILD,
    AgeRange.ADOLESCENT,
    AgeRange.ADULT,
    AgeRange.ELDERLY,
]

FEATURE_NAMES: tuple[str, ...] = (
    *(f"symptom__{symptom.value}" for symptom in Symptom),
    "symptom_count",
    "severity_rank",
    "severity_unknown",
    "duration_log_hours",
    "duration_unknown",
    "age_range_rank",
    "age_unknown",
    *(f"risk__{risk.value}" for risk in RiskFactor),
)


def build_features(extraction: SymptomExtraction, context: PatientContext) -> list[float]:
    present = set(extraction.symptoms)
    age_range = context.age_range
    if age_range is AgeRange.UNKNOWN:
        age_range = AgeRange.from_age(extraction.age)
    severity_known = extraction.severity is not Severity.UNKNOWN
    duration = extraction.duration_minutes

    features = [
        *(float(symptom in present) for symptom in Symptom),
        float(len(present)),
        float(extraction.severity.rank) if severity_known else 0.0,
        float(not severity_known),
        math.log1p(duration / 60) if duration is not None else 0.0,
        float(duration is None),
        float(_AGE_RANGE_ORDER.index(age_range)) if age_range is not AgeRange.UNKNOWN else 0.0,
        float(age_range is AgeRange.UNKNOWN),
        *(float(risk in context.risk_factors) for risk in RiskFactor),
    ]
    assert len(features) == len(FEATURE_NAMES)  # noqa: S101 - programming invariant
    return features
