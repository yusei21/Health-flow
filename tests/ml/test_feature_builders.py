import math

from app.ml.data.schemas import RoutingTrainingExample
from app.ml.feature_builders import (
    FEATURE_BUILDERS,
    HealthFlowSymptomFeatureBuilder,
    MimicStructuredFeatureBuilder,
    TriagegeistStructuredFeatureBuilder,
)
from app.ml.features import build_features
from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange, PatientContext, RiskFactor
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction


def test_structured_builder_uses_vitals_in_declared_order_with_nan_for_missing() -> None:
    example = RoutingTrainingExample(
        label=CareLevel.URGENT_CARE,
        group_id="g",
        heart_rate=90,
        respiratory_rate=None,
        oxygen_saturation=97,
        systolic_bp=120,
        diastolic_bp=80,
        temperature_celsius=37.5,
        pain=0,
        chief_complaint="FAKE",
    )
    values = MimicStructuredFeatureBuilder().build(example)
    assert values[:1] == [90] and math.isnan(values[1])
    assert values[2:] == [97, 120, 80, 37.5, 0]  # pain 0 stays 0, not "missing"
    assert len(values) == len(MimicStructuredFeatureBuilder.feature_names)


def test_structured_builder_never_uses_ids_or_free_text() -> None:
    names = MimicStructuredFeatureBuilder.feature_names
    assert not {"group_id", "subject_id", "stay_id", "chief_complaint"} & set(names)


def test_symptom_builder_matches_api_feature_code() -> None:
    example = RoutingTrainingExample(
        label=CareLevel.PRIMARY_CARE,
        symptoms=[Symptom.COUGH],
        severity=Severity.MILD,
        duration_minutes=600,
        age_range=AgeRange.ADULT,
        risk_factors=[RiskFactor.DIABETES],
    )
    expected = build_features(
        SymptomExtraction(symptoms=[Symptom.COUGH], severity=Severity.MILD, duration_minutes=600),
        PatientContext(age_range=AgeRange.ADULT, risk_factors=frozenset({RiskFactor.DIABETES})),
    )
    assert HealthFlowSymptomFeatureBuilder().build(example) == expected


def test_triagegeist_builder_adds_ordinal_age_and_keeps_missing_as_nan() -> None:
    example = RoutingTrainingExample(
        label=CareLevel.EMERGENCY,
        group_id="g",
        heart_rate=120,
        pain=0,
        age_range=AgeRange.ELDERLY,
        chief_complaint="FAKE",
    )
    builder = TriagegeistStructuredFeatureBuilder()
    values = builder.build(example)
    assert len(values) == len(builder.feature_names)
    assert builder.feature_names[-1] == "age_range_rank"
    assert values[0] == 120 and values[6] == 0  # pain 0 stays 0
    assert values[-1] == 4.0  # INFANT=0 ... ELDERLY=4
    assert all(math.isnan(v) for v in values[1:6])
    unknown = example.model_copy(update={"age_range": AgeRange.UNKNOWN})
    assert math.isnan(builder.build(unknown)[-1])


def test_triagegeist_builder_never_uses_ids_or_free_text() -> None:
    names = TriagegeistStructuredFeatureBuilder.feature_names
    assert not {"group_id", "patient_id", "visit_id", "chief_complaint"} & set(names)


def test_mimic_feature_set_is_unchanged() -> None:
    assert MimicStructuredFeatureBuilder.feature_names == (
        "heart_rate",
        "respiratory_rate",
        "oxygen_saturation",
        "systolic_bp",
        "diastolic_bp",
        "temperature_celsius",
        "pain",
    )


def test_registry_has_one_builder_per_feature_set() -> None:
    assert set(FEATURE_BUILDERS) == {
        "healthflow-symptoms-v1",
        "mimic-structured-vitals-v1",
        "triagegeist-structured-v1",
    }
