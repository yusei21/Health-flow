from uuid import uuid4

from app.context.builder import PatientContextBuilder
from app.repositories.patients import synthetic_demo_patient
from app.schemas.patient import AgeRange, RiskFactor
from app.schemas.symptoms import Symptom, SymptomExtraction

builder = PatientContextBuilder()
record = synthetic_demo_patient(uuid4())


def test_context_excludes_identity_and_encounter_history() -> None:
    context = builder.build(record, SymptomExtraction(symptoms=[Symptom.COUGH]))
    dumped = context.model_dump_json()
    assert record.display_name not in dumped
    assert "Consulta de rotina" not in dumped
    assert context.age_range is AgeRange.ELDERLY
    assert context.risk_factors == {RiskFactor.CARDIOVASCULAR_DISEASE, RiskFactor.DIABETES}


def test_allergies_and_medications_only_when_relevant() -> None:
    unrelated = builder.build(record, SymptomExtraction(symptoms=[Symptom.COUGH]))
    assert unrelated.relevant_allergies == []
    assert unrelated.active_medications == []

    rash = builder.build(record, SymptomExtraction(symptoms=[Symptom.RASH]))
    assert rash.relevant_allergies == ["dipirona"]

    bleeding = builder.build(record, SymptomExtraction(symptoms=[Symptom.SEVERE_BLEEDING]))
    assert bleeding.active_medications == ["varfarina"]


def test_without_record_uses_reported_age_only() -> None:
    context = builder.build(None, SymptomExtraction(age=8))
    assert context.age_range is AgeRange.CHILD
    assert context.risk_factors == frozenset()
