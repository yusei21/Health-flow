"""FAKE `train.csv` files shaped like a Triagegeist export.

Every value is randomly generated. This is NOT Kaggle data and contains no real patient.
The column names are illustrative only (the real header was not available when this was
written); the loader detects columns by alias, so tests also cover alternative names.
Includes messy cases: blanks, text values, invalid acuity, implausible vitals, both
temperature units and repeated visits of the same patient.
"""

import csv
import random
from pathlib import Path

TRAIN_COLUMNS = (
    "patient_id",
    "visit_id",
    "age",
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "systolic_bp",
    "diastolic_bp",
    "temperature",
    "pain_score",
    "arrival_mode",
    "chief_complaint",
    "triage_acuity",
)


def fake_train_rows(
    n_rows: int, n_patients: int | None, seed: int = 0, invalid_rate: float = 0.02
) -> list[dict[str, str]]:
    """`n_patients=None` produces a file WITHOUT a patient identifier column."""
    rng = random.Random(seed)  # noqa: S311 - fake fixture data
    rows = []
    for visit in range(n_rows):
        acuity = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5])
        sick = acuity <= 2
        celsius = rng.gauss(38.4 if sick else 36.9, 0.6)
        temperature = celsius if rng.random() < 0.8 else celsius * 9 / 5 + 32
        row = {
            "patient_id": "" if n_patients is None else f"P{rng.randrange(n_patients):06d}",
            "visit_id": f"V{visit:07d}",
            "age": _maybe(rng, str(rng.randint(0, 95))),
            "heart_rate": _maybe(rng, f"{rng.gauss(115 if sick else 82, 12):.0f}"),
            "respiratory_rate": _maybe(rng, f"{rng.gauss(24 if sick else 16, 3):.0f}"),
            "spo2": _maybe(rng, f"{min(100, rng.gauss(91 if sick else 98, 2)):.0f}"),
            "systolic_bp": _maybe(rng, f"{rng.gauss(150 if sick else 125, 15):.0f}"),
            "diastolic_bp": _maybe(rng, f"{rng.gauss(90 if sick else 78, 10):.0f}"),
            "temperature": _maybe(rng, f"{temperature:.1f}"),
            "pain_score": rng.choice([str(rng.randint(0, 10))] * 8 + ["", "n/a"]),
            "arrival_mode": rng.choice(["walk-in", "ambulance"]),
            "chief_complaint": "FAKE complaint",
            "triage_acuity": (
                str(acuity) if rng.random() > invalid_rate else rng.choice(["", "0", "6", "2.5"])
            ),
        }
        if n_patients is None:
            del row["patient_id"]
        rows.append(row)
    return rows


def _maybe(rng: random.Random, value: str) -> str:
    return "" if rng.random() < 0.05 else value


def write_fake_train(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else TRAIN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path
