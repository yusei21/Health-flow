"""ACADEMIC / SYNTHETIC DATA — NOT FOR CLINICAL USE.

Generates a reproducible synthetic routing dataset. Labels come from a hand-written
generative process, so model metrics measure how well the model recovers that process,
NOT clinical accuracy. Any real dataset can replace it as long as it follows the CSV
schema in `CSV_COLUMNS` (see `load_dataset`).
"""

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange, PatientContext, RiskFactor
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction

logger = logging.getLogger(__name__)

DATA_DISCLAIMER = "ACADEMIC / SYNTHETIC DATA — NOT FOR CLINICAL USE"
DATASET_VERSION = "synthetic-v1"
DEFAULT_DATASET_PATH = Path("data/processed/routing_synthetic_v1.csv")
CSV_COLUMNS = ("symptoms", "severity", "duration_minutes", "age_range", "risk_factors", "label")
_LIST_SEPARATOR = "|"

S = Symptom


@dataclass(frozen=True)
class _Profile:
    pool: tuple[Symptom, ...]
    severities: tuple[Severity, ...]
    duration_hours: tuple[float, float]


_PROFILES: dict[CareLevel, _Profile] = {
    CareLevel.PRIMARY_CARE: _Profile(
        pool=(
            S.COUGH,
            S.SORE_THROAT,
            S.RUNNY_NOSE,
            S.HEADACHE,
            S.BACK_PAIN,
            S.RASH,
            S.FATIGUE,
            S.DIARRHEA,
            S.FEVER,
            S.ABDOMINAL_PAIN,
        ),
        severities=(Severity.MILD, Severity.MILD, Severity.MODERATE, Severity.UNKNOWN),
        duration_hours=(24, 24 * 30),
    ),
    CareLevel.URGENT_CARE: _Profile(
        pool=(
            S.FEVER,
            S.VOMITING,
            S.ABDOMINAL_PAIN,
            S.DIZZINESS,
            S.INJURY,
            S.HEADACHE,
            S.DIARRHEA,
            S.SHORTNESS_OF_BREATH,
            S.CHEST_PAIN,
            S.COUGH,
        ),
        severities=(Severity.MODERATE, Severity.MODERATE, Severity.SEVERE, Severity.UNKNOWN),
        duration_hours=(1, 72),
    ),
    CareLevel.EMERGENCY: _Profile(
        pool=(
            S.CHEST_PAIN,
            S.SHORTNESS_OF_BREATH,
            S.LOSS_OF_CONSCIOUSNESS,
            S.SEIZURE,
            S.FACIAL_DROOP,
            S.SLURRED_SPEECH,
            S.ONE_SIDED_WEAKNESS,
            S.SEVERE_BLEEDING,
            S.ABDOMINAL_PAIN,
            S.VOMITING,
            S.DIZZINESS,
        ),
        severities=(Severity.SEVERE, Severity.SEVERE, Severity.MODERATE, Severity.UNKNOWN),
        duration_hours=(0.05, 12),
    ),
}
_CLASS_PRIOR = {
    CareLevel.PRIMARY_CARE: 0.55,
    CareLevel.URGENT_CARE: 0.30,
    CareLevel.EMERGENCY: 0.15,
}
_LABEL_NOISE = 0.05


@dataclass(frozen=True)
class RoutingExample:
    extraction: SymptomExtraction
    context: PatientContext
    label: CareLevel


def generate_examples(n: int, seed: int) -> list[RoutingExample]:
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data, not security
    return [_generate_one(rng) for _ in range(n)]


def _generate_one(rng: random.Random) -> RoutingExample:
    label = rng.choices(list(_CLASS_PRIOR), weights=list(_CLASS_PRIOR.values()))[0]
    profile = _PROFILES[label]
    low, high = profile.duration_hours

    symptoms = rng.sample(profile.pool, k=rng.randint(1, 3))
    if rng.random() < 0.3:  # cross-profile noise so classes overlap
        symptoms.append(rng.choice(list(Symptom)))
    age_range = rng.choice([r for r in AgeRange if r is not AgeRange.UNKNOWN] + [AgeRange.UNKNOWN])
    risks = frozenset(r for r in RiskFactor if rng.random() < _risk_probability(r, age_range))
    duration = None if rng.random() < 0.15 else round(rng.uniform(low, high) * 60)

    # Vulnerable patients with urgent-looking reports are sometimes escalated.
    if label is CareLevel.URGENT_CARE and risks and age_range is AgeRange.ELDERLY:
        label = CareLevel.EMERGENCY if rng.random() < 0.4 else label
    if rng.random() < _LABEL_NOISE:
        label = rng.choice(list(CareLevel))

    return RoutingExample(
        extraction=SymptomExtraction(
            symptoms=symptoms, severity=rng.choice(profile.severities), duration_minutes=duration
        ),
        context=PatientContext(age_range=age_range, risk_factors=risks),
        label=label,
    )


def _risk_probability(risk: RiskFactor, age_range: AgeRange) -> float:
    if risk is RiskFactor.PREGNANCY:
        return 0.05 if age_range in {AgeRange.ADULT, AgeRange.ADOLESCENT} else 0.0
    return 0.25 if age_range is AgeRange.ELDERLY else 0.07


def save_dataset(examples: list[RoutingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for ex in examples:
            writer.writerow(
                [
                    _LIST_SEPARATOR.join(s.value for s in ex.extraction.symptoms),
                    ex.extraction.severity.value,
                    ""
                    if ex.extraction.duration_minutes is None
                    else ex.extraction.duration_minutes,
                    ex.context.age_range.value,
                    _LIST_SEPARATOR.join(sorted(r.value for r in ex.context.risk_factors)),
                    ex.label.value,
                ]
            )
    path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "disclaimer": DATA_DISCLAIMER,
                "dataset_version": DATASET_VERSION,
                "rows": len(examples),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_dataset(path: Path) -> list[RoutingExample]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"dataset columns must be {CSV_COLUMNS}")
        return [_parse_row(row) for row in reader]


def _parse_row(row: dict[str, str]) -> RoutingExample:
    def split(value: str) -> list[str]:
        return [item for item in value.split(_LIST_SEPARATOR) if item]

    return RoutingExample(
        extraction=SymptomExtraction(
            symptoms=[Symptom(code) for code in split(row["symptoms"])],
            severity=Severity(row["severity"]),
            duration_minutes=int(row["duration_minutes"]) if row["duration_minutes"] else None,
        ),
        context=PatientContext(
            age_range=AgeRange(row["age_range"]),
            risk_factors=frozenset(RiskFactor(code) for code in split(row["risk_factors"])),
        ),
        label=CareLevel(row["label"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {DATA_DISCLAIMER}")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/routing_synthetic_v1.csv")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    save_dataset(generate_examples(args.rows, args.seed), args.output)
    logger.info("dataset written: %s (%s rows) — %s", args.output, args.rows, DATA_DISCLAIMER)


if __name__ == "__main__":
    main()
