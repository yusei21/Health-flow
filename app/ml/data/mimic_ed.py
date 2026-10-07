"""MIMIC-IV-ED `triage` table → canonical training examples.

MIMIC-IV-ED is credentialed PhysioNet data: it is never downloaded, committed or
redistributed by this project. The user places the file locally (see docs/datasets.md).

Cleaning rules (all counted in `PreparationReport`, never silent):
- rows with acuity missing or outside 1-5 are dropped (no class is invented);
- rows without subject_id are dropped (the patient-level split needs it);
- vitals outside broad physical-plausibility bounds become missing (None); these bounds
  are data-quality guards, NOT clinical thresholds;
- pain is kept only when it is a plain number in 0-10; free text ("unable", "denies")
  becomes missing;
- temperature is converted from °F (MIMIC unit) to °C; no unit guessing is attempted.
Missing values are left as None here; imputation happens inside the sklearn Pipeline.
"""

import csv
import hashlib
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.data.parsing import open_text, parse_number
from app.ml.data.schemas import DatasetInfo, RoutingTrainingExample
from app.schemas.care import CareLevel

MIMIC_DISCLAIMER = (
    "MIMIC-IV-ED (US emergency departments). Credentialed data: do not redistribute. "
    "Experimental ESI→3-class mapping; NOT an SUS protocol and NOT clinical validation."
)
ESI_MAPPING_DESCRIPTION = (
    "Experimental Health-flow simplification: ESI 1-2 → EMERGENCY, ESI 3 → URGENT_CARE, "
    "ESI 4-5 → PRIMARY_CARE. Created only to compare three classes; ESI 4-5 is not "
    "clinically equivalent to UBS/SUS primary care."
)
MIMIC_DATASET_VERSION = "mimic-ed-v1"
TRIAGE_FILENAMES = ("triage.csv.gz", "triage.csv")
REQUIRED_COLUMNS = frozenset({"subject_id", "acuity"})

_ESI_TO_CARE_LEVEL = {
    1: CareLevel.EMERGENCY,
    2: CareLevel.EMERGENCY,
    3: CareLevel.URGENT_CARE,
    4: CareLevel.PRIMARY_CARE,
    5: CareLevel.PRIMARY_CARE,
}


@dataclass(frozen=True)
class _Vital:
    source_column: str
    target_field: str
    low: float
    high: float


# Broad data-quality bounds in the source unit; values outside are recording errors.
_VITALS = (
    _Vital("heartrate", "heart_rate", 20, 300),
    _Vital("resprate", "respiratory_rate", 4, 80),
    _Vital("o2sat", "oxygen_saturation", 50, 100),
    _Vital("sbp", "systolic_bp", 40, 300),
    _Vital("dbp", "diastolic_bp", 20, 200),
    _Vital("temperature", "temperature_celsius", 86, 113),  # °F
)


def mimic_dataset_info(source_version: str) -> DatasetInfo:
    return DatasetInfo(
        name="mimic-iv-ed",
        version=f"{MIMIC_DATASET_VERSION}+source-{source_version}",
        disclaimer=MIMIC_DISCLAIMER,
        label_definition=ESI_MAPPING_DESCRIPTION,
    )


def map_esi_to_care_level(acuity: object) -> CareLevel | None:
    """ESI acuity (1-5) → CareLevel; None when missing or invalid (row must be dropped)."""
    if isinstance(acuity, bool) or acuity is None:
        return None
    if isinstance(acuity, str):
        acuity = parse_number(acuity)
    if not isinstance(acuity, int | float) or not math.isfinite(acuity):
        return None
    if acuity != int(acuity):
        return None
    return _ESI_TO_CARE_LEVEL.get(int(acuity))


def pseudonymous_group_id(subject_id: str) -> str:
    """Stable per-patient key for group splitting; pseudonymization, not anonymization."""
    return hashlib.sha256(f"mimic-iv-ed:{subject_id}".encode()).hexdigest()[:16]


def resolve_triage_path(path: Path) -> Path:
    """Accept a directory, `triage.csv.gz` or `triage.csv` (falling back to the other)."""
    if path.is_dir():
        candidates = [path / name for name in TRIAGE_FILENAMES]
    elif path.name.endswith(".csv.gz"):
        candidates = [path, path.with_name(path.name.removesuffix(".gz"))]
    elif path.suffix == ".csv":
        candidates = [path, path.with_name(path.name + ".gz")]
    else:
        candidates = [path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"MIMIC-IV-ED triage file not found (tried: {', '.join(str(c) for c in candidates)}). "
        "See docs/datasets.md."
    )


@dataclass
class PreparationReport:
    """Aggregate-only statistics; safe to log (no identifiers, no free text)."""

    rows_read: int = 0
    rows_kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    missing: Counter[str] = field(default_factory=Counter)
    implausible: Counter[str] = field(default_factory=Counter)
    pain_invalid: int = 0
    absent_columns: list[str] = field(default_factory=list)
    class_distribution: Counter[str] = field(default_factory=Counter)
    patients: set[str] = field(default_factory=set, repr=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_dropped": dict(self.dropped),
            "missing_after_cleaning": dict(self.missing),
            "implausible_set_to_missing": dict(self.implausible),
            "pain_invalid_set_to_missing": self.pain_invalid,
            "absent_columns": self.absent_columns,
            "class_distribution": dict(self.class_distribution),
            "number_of_patients": len(self.patients),
        }


def read_triage(path: Path, report: PreparationReport) -> Iterator[RoutingTrainingExample]:
    """Stream canonical examples from a triage CSV or CSV.gz, filling `report`."""
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        if missing := REQUIRED_COLUMNS - columns:
            raise ValueError(f"triage file lacks required columns: {sorted(missing)}")
        optional = {v.source_column for v in _VITALS} | {"pain", "chiefcomplaint"}
        report.absent_columns = sorted(optional - columns)
        for row in reader:
            report.rows_read += 1
            example = _convert_row(row, report)
            if example is not None:
                report.rows_kept += 1
                report.class_distribution[example.label.value] += 1
                yield example


def _convert_row(row: dict[str, str], report: PreparationReport) -> RoutingTrainingExample | None:
    label = map_esi_to_care_level(row.get("acuity"))
    if label is None:
        report.dropped["invalid_or_missing_acuity"] += 1
        return None
    subject_id = (row.get("subject_id") or "").strip()
    if not subject_id:
        report.dropped["missing_subject_id"] += 1
        return None
    group_id = pseudonymous_group_id(subject_id)
    report.patients.add(group_id)

    values: dict[str, float | None] = {
        vital.target_field: _clean_vital(row.get(vital.source_column), vital, report)
        for vital in _VITALS
    }
    temperature_f = values["temperature_celsius"]
    if temperature_f is not None:
        values["temperature_celsius"] = round((temperature_f - 32) * 5 / 9, 2)
    values["pain"] = _clean_pain(row.get("pain"), report)
    for name, value in values.items():
        if value is None:
            report.missing[name] += 1

    complaint = (row.get("chiefcomplaint") or "").strip() or None
    return RoutingTrainingExample(
        group_id=group_id, label=label, chief_complaint=complaint, **values
    )


def _clean_vital(raw: str | None, vital: _Vital, report: PreparationReport) -> float | None:
    value = parse_number(raw)
    if value is None:
        return None
    if not vital.low <= value <= vital.high:
        report.implausible[vital.target_field] += 1
        return None
    return value


def _clean_pain(raw: str | None, report: PreparationReport) -> float | None:
    if raw is None or not raw.strip():
        return None
    value = parse_number(raw)
    if value is None or not 0 <= value <= 10:
        report.pain_invalid += 1
        return None
    return value
