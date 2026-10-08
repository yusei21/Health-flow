"""Context-aware follow-up prompts for an experimental navigation interface.

This module does not diagnose, infer severity, or suppress emergency safety checks.
It only avoids re-asking details that are explicitly present in the original report.
"""

import re

from app.safety.engine import normalize_text
from app.schemas.symptoms import SymptomExtraction

_DURATION = re.compile(
    r"\b(hoje|ontem|anteontem|agora|desde (?:hoje|ontem|anteontem)|"
    r"ha \d+ (?:minutos?|horas?|dias?|semanas?|meses?)|"
    r"comecou (?:hoje|ontem|anteontem)|"
    r"desde (?:a|as) \d+ (?:horas?|dias?))\b"
)
_ACCIDENT = re.compile(
    r"\b(acidente|queda|cai|caiu|bati|batida|trauma|ferimento|machuquei)\b"
)
_ACCIDENT_DENIED = re.compile(
    r"\b(?:nao (?:houve|tive|sofri|aconteceu)|sem) (?:nenhum )?"
    r"(?:acidente|queda|trauma|ferimento)\b"
)
_SEVERITY = re.compile(
    r"\b(?:dor|sintoma|doendo|doi) (?:muito |bem )?"
    r"(?:leve|fraca|moderada|forte|intensa|insuportavel)\b"
    r"|\b(?:dor|sintoma) de (?:intensidade )?"
    r"(?:leve|moderada|forte|intensa)\b"
    r"|\b(?:leve|moderada|forte|intensa|insuportavel)\b"
)
_EMERGENCY_SIGNS = re.compile(
    r"\b(?:dor no peito|falta de ar|desmaio|desmaiei|"
    r"sangue nas fezes|vomitando sangue|convulsao)\b"
)


def follow_up_questions(message: str, extraction: SymptomExtraction | None) -> list[str]:
    """Ask for missing information without assuming that unmentioned signs are absent."""
    report = normalize_text(message)
    questions: list[str] = []

    if not (_DURATION.search(report) or (
        extraction is not None and extraction.duration_minutes is not None
    )):
        questions.append("Quando começaram os sintomas?")

    if not (_SEVERITY.search(report) or (
        extraction is not None and extraction.severity.value != "unknown"
    )):
        questions.append("Qual é a intensidade da dor ou do sintoma (leve, moderada ou forte)?")

    if not _EMERGENCY_SIGNS.search(report):
        questions.append(
            "Há dor no peito, falta de ar, desmaio, sangue nas fezes ou outro sinal de gravidade?"
        )

    if not (_ACCIDENT.search(report) or _ACCIDENT_DENIED.search(report)):
        questions.append("Houve queda, acidente ou ferimento?")

    return questions
