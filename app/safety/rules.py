"""ACADEMIC / SIMULATED safety rules.

These are illustrative red flags written for the prototype. They are NOT an official
SUS or Manchester protocol and must be replaced by clinically validated rules before
any real-world use.

Text patterns are matched against lowercase, accent-free text and deliberately
over-trigger (e.g. they ignore negation): a false alarm is preferable to a miss.
"""

from app.safety.schemas import SafetyRule
from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange
from app.schemas.symptoms import Severity, Symptom

EMERGENCY = CareLevel.EMERGENCY
URGENT = CareLevel.URGENT_CARE

ACADEMIC_SAFETY_RULES: tuple[SafetyRule, ...] = (
    SafetyRule(
        id="RED_FLAG_001",
        description="Chest pain together with shortness of breath",
        minimum_care_level=EMERGENCY,
        all_symptoms=frozenset({Symptom.CHEST_PAIN, Symptom.SHORTNESS_OF_BREATH}),
    ),
    SafetyRule(
        id="RED_FLAG_002",
        description="Loss of consciousness / fainting",
        minimum_care_level=EMERGENCY,
        any_symptoms=frozenset({Symptom.LOSS_OF_CONSCIOUSNESS}),
        text_patterns=("desmai", "perdi a consciencia", "perdeu a consciencia", "inconsciente"),
    ),
    SafetyRule(
        id="RED_FLAG_003",
        description="Seizure",
        minimum_care_level=EMERGENCY,
        any_symptoms=frozenset({Symptom.SEIZURE}),
        text_patterns=("convuls",),
    ),
    SafetyRule(
        id="RED_FLAG_004",
        description="Possible neurological deficit signs (face, speech, one side of body)",
        minimum_care_level=EMERGENCY,
        any_symptoms=frozenset(
            {Symptom.FACIAL_DROOP, Symptom.SLURRED_SPEECH, Symptom.ONE_SIDED_WEAKNESS}
        ),
        text_patterns=("boca torta", "rosto torto", "fala enrolada", "fala arrastada"),
    ),
    SafetyRule(
        id="RED_FLAG_005",
        description="Severe bleeding",
        minimum_care_level=EMERGENCY,
        any_symptoms=frozenset({Symptom.SEVERE_BLEEDING}),
        text_patterns=("sangramento intenso", "sangrando muito", "vomitando sangue"),
    ),
    SafetyRule(
        id="RED_FLAG_006",
        description="Severe shortness of breath",
        minimum_care_level=EMERGENCY,
        any_symptoms=frozenset({Symptom.SHORTNESS_OF_BREATH}),
        min_severity=Severity.SEVERE,
        text_patterns=("nao consigo respirar",),
    ),
    SafetyRule(
        id="RED_FLAG_007",
        description="Self-harm ideation",
        minimum_care_level=EMERGENCY,
        any_symptoms=frozenset({Symptom.SELF_HARM_IDEATION}),
        text_patterns=("suicid", "me matar", "tirar minha vida", "acabar com minha vida"),
    ),
    SafetyRule(
        id="CAUTION_001",
        description="Chest pain without other red flags",
        minimum_care_level=URGENT,
        any_symptoms=frozenset({Symptom.CHEST_PAIN}),
    ),
    SafetyRule(
        id="CAUTION_002",
        description="Severe fever",
        minimum_care_level=URGENT,
        any_symptoms=frozenset({Symptom.FEVER}),
        min_severity=Severity.SEVERE,
    ),
    SafetyRule(
        id="CAUTION_003",
        description="Fever in infants or elderly people",
        minimum_care_level=URGENT,
        any_symptoms=frozenset({Symptom.FEVER}),
        age_ranges=frozenset({AgeRange.INFANT, AgeRange.ELDERLY}),
    ),
)
