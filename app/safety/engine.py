import unicodedata
from collections.abc import Iterable

from app.safety.rules import ACADEMIC_SAFETY_RULES
from app.safety.schemas import SafetyAssessment, SafetyRule
from app.schemas.care import CareLevel, most_severe
from app.schemas.patient import AgeRange, PatientContext
from app.schemas.symptoms import SymptomExtraction


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_accents.split())


class SafetyEngine:
    """Deterministic red-flag checks. Independent of the LLM's and ML's opinions."""

    def __init__(self, rules: Iterable[SafetyRule] = ACADEMIC_SAFETY_RULES) -> None:
        self._rules = tuple(rules)
        ids = [rule.id for rule in self._rules]
        if len(ids) != len(set(ids)):
            raise ValueError("safety rule ids must be unique")

    def assess(
        self,
        report_text: str,
        extraction: SymptomExtraction | None,
        context: PatientContext | None,
    ) -> SafetyAssessment:
        normalized = normalize_text(report_text)
        age_range = self._age_range(extraction, context)
        matched = [
            rule
            for rule in self._rules
            if self._matches_text(rule, normalized)
            or (extraction is not None and self._matches_structured(rule, extraction, age_range))
        ]
        minimum = most_severe(*(rule.minimum_care_level for rule in matched))
        return SafetyAssessment(
            has_red_flag=minimum is CareLevel.EMERGENCY,
            matched_rules=[rule.id for rule in matched],
            suggested_minimum_care_level=minimum,
        )

    @staticmethod
    def _age_range(
        extraction: SymptomExtraction | None, context: PatientContext | None
    ) -> AgeRange:
        if context is not None and context.age_range is not AgeRange.UNKNOWN:
            return context.age_range
        return AgeRange.from_age(extraction.age if extraction else None)

    @staticmethod
    def _matches_text(rule: SafetyRule, normalized_text: str) -> bool:
        return any(
            pattern in normalized_text
            and not _negated_pattern(normalized_text, pattern)
            for pattern in rule.text_patterns
        )

    @staticmethod
    def _matches_structured(
        rule: SafetyRule, extraction: SymptomExtraction, age_range: AgeRange
    ) -> bool:
        if not (rule.all_symptoms or rule.any_symptoms):
            return False
        present = set(extraction.symptoms)
        if not rule.all_symptoms <= present:
            return False
        if rule.any_symptoms and not rule.any_symptoms & present:
            return False
        if rule.min_severity is not None and extraction.severity.rank < rule.min_severity.rank:
            return False
        return not rule.age_ranges or age_range in rule.age_ranges


def _negated_pattern(text: str, pattern: str) -> bool:
    """Conservative negation screen for common explicit statements in Portuguese.

    Ambiguous phrases still require clinical review; no substring logic is clinically
    reliable enough to replace professional assessment.
    """
    start = text.find(pattern)
    while start >= 0:
        prefix = text[max(0, start - 35):start].strip()
        if not any(prefix.endswith(phrase) for phrase in (
            "nao", "nunca", "nem", "sem", "nao tive", "nao estou",
            "nao ha", "nego", "sem sinais de",
        )):
            return False
        start = text.find(pattern, start + len(pattern))
    return True
