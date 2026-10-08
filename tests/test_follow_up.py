"""Regression tests for contextual questions in colloquial Brazilian Portuguese."""

from app.harness.follow_up import follow_up_questions
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction


def test_does_not_repeat_known_timing_or_denied_accident() -> None:
    report = (
        "estou com dor de barriga comecou hoje so dor de barriga e diarreia "
        "nao houve acidente esta doendo quando defeco"
    )
    extraction = SymptomExtraction(
        symptoms=[Symptom.ABDOMINAL_PAIN, Symptom.DIARRHEA],
        severity=Severity.UNKNOWN,
    )
    questions = follow_up_questions(report, extraction)
    assert not any("Quando começaram" in question for question in questions)
    assert not any("acidente" in question for question in questions)
    assert any("intensidade" in question for question in questions)
    assert any("sangue nas fezes" in question for question in questions)


def test_unknown_details_are_asked_not_assumed_absent() -> None:
    questions = follow_up_questions("dor no ombro", SymptomExtraction())
    assert any("Quando" in question for question in questions)
    assert any("intensidade" in question for question in questions)
    assert any("sinal de gravidade" in question for question in questions)


def test_structured_duration_and_severity_prevent_duplicate_questions() -> None:
    extraction = SymptomExtraction(duration_minutes=120, severity=Severity.MODERATE)
    questions = follow_up_questions("dor de barriga", extraction)
    assert not any("Quando" in question for question in questions)
    assert not any("intensidade" in question for question in questions)
