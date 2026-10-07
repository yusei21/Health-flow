import math

from app.ml.features import FEATURE_NAMES, build_features
from app.schemas.patient import AgeRange, PatientContext, RiskFactor
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction


def features_by_name(extraction: SymptomExtraction, context: PatientContext) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, build_features(extraction, context), strict=True))


def test_encodes_symptoms_severity_duration_age_and_risks() -> None:
    f = features_by_name(
        SymptomExtraction(
            symptoms=[Symptom.FEVER, Symptom.COUGH], severity=Severity.SEVERE, duration_minutes=120
        ),
        PatientContext(age_range=AgeRange.ELDERLY, risk_factors=frozenset({RiskFactor.DIABETES})),
    )
    assert f["symptom__fever"] == 1 and f["symptom__cough"] == 1 and f["symptom__rash"] == 0
    assert f["symptom_count"] == 2
    assert f["severity_rank"] == Severity.SEVERE.rank and f["severity_unknown"] == 0
    assert f["duration_log_hours"] == math.log1p(2) and f["duration_unknown"] == 0
    assert f["age_range_rank"] == 4 and f["age_unknown"] == 0
    assert f["risk__diabetes"] == 1 and f["risk__pregnancy"] == 0


def test_unknown_values_have_explicit_indicator_features() -> None:
    f = features_by_name(SymptomExtraction(), PatientContext())
    assert f["severity_unknown"] == 1
    assert f["duration_unknown"] == 1
    assert f["age_unknown"] == 1


def test_falls_back_to_reported_age_when_record_has_none() -> None:
    f = features_by_name(SymptomExtraction(age=5), PatientContext())
    assert f["age_range_rank"] == 1 and f["age_unknown"] == 0
