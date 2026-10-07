import pytest

from app.safety.engine import SafetyEngine, normalize_text
from app.safety.rules import ACADEMIC_SAFETY_RULES
from app.safety.schemas import SafetyRule
from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange, PatientContext
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction

engine = SafetyEngine()


def assess(
    text: str = "",
    symptoms: list[Symptom] | None = None,
    severity: Severity = Severity.UNKNOWN,
    age_range: AgeRange = AgeRange.ADULT,
    with_extraction: bool = True,
) -> tuple[bool, list[str], CareLevel | None]:
    extraction = (
        SymptomExtraction(symptoms=symptoms or [], severity=severity) if with_extraction else None
    )
    result = engine.assess(text, extraction, PatientContext(age_range=age_range))
    return result.has_red_flag, result.matched_rules, result.suggested_minimum_care_level


@pytest.mark.parametrize(
    ("rule_id", "symptoms", "severity"),
    [
        ("RED_FLAG_001", [Symptom.CHEST_PAIN, Symptom.SHORTNESS_OF_BREATH], Severity.MILD),
        ("RED_FLAG_002", [Symptom.LOSS_OF_CONSCIOUSNESS], Severity.UNKNOWN),
        ("RED_FLAG_003", [Symptom.SEIZURE], Severity.UNKNOWN),
        ("RED_FLAG_004", [Symptom.FACIAL_DROOP], Severity.UNKNOWN),
        ("RED_FLAG_004", [Symptom.SLURRED_SPEECH], Severity.UNKNOWN),
        ("RED_FLAG_004", [Symptom.ONE_SIDED_WEAKNESS], Severity.UNKNOWN),
        ("RED_FLAG_005", [Symptom.SEVERE_BLEEDING], Severity.UNKNOWN),
        ("RED_FLAG_006", [Symptom.SHORTNESS_OF_BREATH], Severity.SEVERE),
        ("RED_FLAG_007", [Symptom.SELF_HARM_IDEATION], Severity.UNKNOWN),
    ],
)
def test_each_red_flag_requires_emergency_from_structured_symptoms(
    rule_id: str, symptoms: list[Symptom], severity: Severity
) -> None:
    red_flag, matched, minimum = assess(symptoms=symptoms, severity=severity)
    assert red_flag
    assert rule_id in matched
    assert minimum is CareLevel.EMERGENCY


@pytest.mark.parametrize(
    ("rule_id", "text"),
    [
        ("RED_FLAG_002", "Meu pai DESMAIOU agora há pouco"),
        ("RED_FLAG_003", "ela está tendo uma convulsão"),
        ("RED_FLAG_004", "acordei com a boca torta"),
        ("RED_FLAG_005", "estou vomitando sangue"),
        ("RED_FLAG_006", "não consigo respirar direito"),
        ("RED_FLAG_007", "tenho pensado em suicídio"),
    ],
)
def test_text_patterns_fire_even_without_llm_extraction(rule_id: str, text: str) -> None:
    red_flag, matched, minimum = assess(text=text, with_extraction=False)
    assert red_flag
    assert rule_id in matched
    assert minimum is CareLevel.EMERGENCY


def test_caution_rules_set_urgent_floor_without_red_flag() -> None:
    red_flag, matched, minimum = assess(symptoms=[Symptom.CHEST_PAIN], severity=Severity.MILD)
    assert not red_flag
    assert matched == ["CAUTION_001"]
    assert minimum is CareLevel.URGENT_CARE


@pytest.mark.parametrize(
    ("age_range", "expected"),
    [(AgeRange.ELDERLY, ["CAUTION_003"]), (AgeRange.INFANT, ["CAUTION_003"]), (AgeRange.ADULT, [])],
)
def test_fever_floor_depends_on_age_range(age_range: AgeRange, expected: list[str]) -> None:
    _, matched, _ = assess(symptoms=[Symptom.FEVER], severity=Severity.MILD, age_range=age_range)
    assert matched == expected


def test_severity_threshold_is_respected() -> None:
    red_flag, matched, _ = assess(
        symptoms=[Symptom.SHORTNESS_OF_BREATH], severity=Severity.MODERATE
    )
    assert not red_flag
    assert "RED_FLAG_006" not in matched


def test_mild_complaint_has_no_floor() -> None:
    assert assess(text="tosse e coriza", symptoms=[Symptom.COUGH, Symptom.RUNNY_NOSE]) == (
        False,
        [],
        None,
    )


def test_explicit_negation_does_not_trigger_fainting_red_flag() -> None:
    red_flag, _, _ = assess(text="não desmaiei, só fiquei tonto", with_extraction=False)
    assert not red_flag

def test_normalize_text_removes_accents_case_and_extra_spaces() -> None:
    assert normalize_text("  Não   CONSIGO respirar ") == "nao consigo respirar"


def test_rule_ids_are_unique_and_every_rule_is_marked_academic() -> None:
    ids = [rule.id for rule in ACADEMIC_SAFETY_RULES]
    assert len(ids) == len(set(ids))
    assert all("ACADEMIC" in rule.source for rule in ACADEMIC_SAFETY_RULES)


def test_engine_rejects_duplicate_rule_ids() -> None:
    rule = ACADEMIC_SAFETY_RULES[0]
    with pytest.raises(ValueError, match="unique"):
        SafetyEngine([rule, rule])


def test_rule_without_trigger_is_invalid() -> None:
    with pytest.raises(ValueError, match="no trigger"):
        SafetyRule(id="BAD_RULE_001", description="x", minimum_care_level=CareLevel.EMERGENCY)
