"""Kaggle Triagegeist `train.csv` → canonical training examples.

The dataset is downloaded MANUALLY by the user from Kaggle and placed in
`data/raw/triagegeist/` (see docs/datasets.md). This module never downloads anything, never
uses Kaggle credentials and the raw files are never committed or redistributed.

Column names are DETECTED, not assumed: every canonical field has an explicit list of
accepted source names (`detect_columns`), compared after normalization (lower case,
non-alphanumerics → "_"). The detected correspondence is written to the processed
dataset metadata. Two source columns matching the same field is an error (no silent pick).

Cleaning rules (all counted in `TriagegeistReport`, never silent):
- target: `triage_acuity` (overridable); missing or outside 1-5 → row dropped, no class
  is invented. Mapping 1-2 → EMERGENCY, 3 → URGENT_CARE, 4-5 → PRIMARY_CARE is an
  experimental Health-flow simplification, NOT an SUS protocol;
- patient identifier, when a column exists: rows without it are dropped (the patient-level
  split needs it) and it is replaced by a pseudonymous hash. When no such column exists,
  `group_id` stays None and the split is row-level: there is NO patient-leakage protection;
- other identifiers (row/visit ids) and unrecognized columns are never features; their names
  are only listed in the report;
- vitals/pain/age outside broad physical-plausibility bounds become missing (None); bounds
  are data-quality guards, NOT clinical thresholds;
- temperature: explicit unit from the column name (`*_c` / `*_f`); for an ambiguous name
  each value is classified by the disjoint plausible ranges 30-45 (°C) and 86-113 (°F),
  anything else becomes missing;
- missing values stay None; imputation happens inside the sklearn Pipeline.
The chief complaint is kept as free text for a future LLM experiment and is never a feature.
"""

import csv
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from app.ml.data.mimic_ed import map_esi_to_care_level
from app.ml.data.parsing import open_text, parse_number
from app.ml.data.schemas import DatasetInfo, RoutingTrainingExample
from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange

TRIAGEGEIST_DATASET_VERSION = "triagegeist-v1"
TRIAGEGEIST_SOURCE = "Kaggle Triagegeist (train.csv), downloaded manually by the user"
TRIAGEGEIST_DISCLAIMER = (
    "Kaggle Triagegeist. Not redistributed by this repository; subject to the Kaggle "
    "dataset/competition terms. Experimental 1-5 → 3-class mapping; NOT an SUS protocol, "
    "NOT representative of the SUS and NOT clinical validation."
)
TRIAGE_ACUITY_MAPPING_DESCRIPTION = (
    "Experimental Health-flow simplification: triage_acuity 1-2 → EMERGENCY, 3 → URGENT_CARE, "
    "4-5 → PRIMARY_CARE. Created only to compare three classes; acuity 4-5 is not clinically "
    "equivalent to UBS/SUS primary care."
)
DEFAULT_TARGET_COLUMN = "triage_acuity"
TRAIN_FILENAME = "train.csv"


def map_triage_acuity_to_care_level(acuity: object) -> CareLevel | None:
    """Triagegeist `triage_acuity` (1-5) → CareLevel; None when missing or invalid.

    Same experimental 1-5 → 3-class table as the MIMIC ESI mapping. It does not claim
    that Triagegeist acuity is ESI, nor that 4-5 equals UBS care.
    """
    return map_esi_to_care_level(acuity)


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


@dataclass(frozen=True)
class _NumericField:
    target_field: str
    aliases: tuple[str, ...]
    low: float
    high: float


# Assumed units: bpm, breaths/min, %, mmHg, 0-10 scale, years.
_NUMERIC_FIELDS = (
    _NumericField("heart_rate", ("heart_rate", "heartrate", "hr", "pulse", "pulse_rate"), 20, 300),
    _NumericField(
        "respiratory_rate",
        ("respiratory_rate", "resp_rate", "resprate", "rr", "respiration_rate"),
        4,
        80,
    ),
    _NumericField(
        "oxygen_saturation",
        ("oxygen_saturation", "o2_saturation", "o2sat", "o2_sat", "spo2", "sao2"),
        50,
        100,
    ),
    _NumericField(
        "systolic_bp",
        ("systolic_bp", "sbp", "systolic_blood_pressure", "bp_systolic", "systolic"),
        40,
        300,
    ),
    _NumericField(
        "diastolic_bp",
        ("diastolic_bp", "dbp", "diastolic_blood_pressure", "bp_diastolic", "diastolic"),
        20,
        200,
    ),
    _NumericField("pain", ("pain", "pain_score", "pain_scale", "pain_level", "nrs_pain"), 0, 10),
    _NumericField("age", ("age", "age_years", "patient_age"), 0, 120),
)


class TemperatureUnit(StrEnum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    PER_VALUE_RANGE = "per_value_range"  # ambiguous name: classify each value by range


_TEMPERATURE_ALIASES: dict[str, TemperatureUnit] = {
    **dict.fromkeys(
        ("temperature_c", "temp_c", "temperature_celsius", "temp_celsius", "body_temperature_c"),
        TemperatureUnit.CELSIUS,
    ),
    **dict.fromkeys(
        ("temperature_f", "temp_f", "temperature_fahrenheit", "temp_fahrenheit"),
        TemperatureUnit.FAHRENHEIT,
    ),
    **dict.fromkeys(("temperature", "temp", "body_temperature"), TemperatureUnit.PER_VALUE_RANGE),
}
_CELSIUS_RANGE = (30.0, 45.0)
_FAHRENHEIT_RANGE = (86.0, 113.0)

_PATIENT_ID_ALIASES = (
    "patient_id",
    "subject_id",
    "patient",
    "patient_key",
    "patient_identifier",
    "person_id",
    "mrn",
)
# Never features; listed only so the report shows they were seen and ignored.
_OTHER_ID_ALIASES = (
    "id",
    "row_id",
    "visit_id",
    "encounter_id",
    "stay_id",
    "triage_id",
    "case_id",
    "admission_id",
    "record_id",
)
_COMPLAINT_ALIASES = (
    "chief_complaint",
    "chiefcomplaint",
    "complaint",
    "presenting_complaint",
    "reason_for_visit",
)
STRUCTURED_FIELDS = (*(f.target_field for f in _NUMERIC_FIELDS), "temperature_celsius")


@dataclass(frozen=True)
class ColumnMapping:
    """Detected correspondence between the source CSV and the canonical schema."""

    target: str
    patient_id: str | None
    chief_complaint: str | None
    numeric: dict[str, str]  # canonical field → source column
    temperature: str | None
    temperature_unit: TemperatureUnit | None
    ignored_identifiers: list[str]
    unused: list[str]

    def as_dict(self) -> dict[str, object]:
        fields: dict[str, str | None] = {name: self.numeric.get(name) for name in STRUCTURED_FIELDS}
        fields["temperature_celsius"] = self.temperature
        return {
            "target": self.target,
            "patient_id": self.patient_id,
            "chief_complaint": self.chief_complaint,
            "canonical_fields": fields,
            "temperature_unit": self.temperature_unit,
            "ignored_identifier_columns": self.ignored_identifiers,
            "unused_columns": self.unused,
        }


def detect_columns(
    fieldnames: Iterable[str], target_column: str = DEFAULT_TARGET_COLUMN
) -> ColumnMapping:
    """Match real CSV headers to canonical fields; fail loudly on missing/ambiguous columns."""
    columns = list(fieldnames)
    by_normalized: dict[str, list[str]] = {}
    for column in columns:
        by_normalized.setdefault(normalize_column(column), []).append(column)
    used: set[str] = set()

    def find(aliases: Iterable[str], role: str) -> str | None:
        matches = [c for alias in aliases for c in by_normalized.get(alias, [])]
        if len(matches) > 1:
            raise ValueError(f"ambiguous source columns for {role}: {sorted(matches)}")
        if matches:
            used.add(matches[0])
            return matches[0]
        return None

    target = find([normalize_column(target_column)], "target")
    if target is None:
        raise ValueError(
            f"Triagegeist file lacks the target column {target_column!r}; "
            f"columns found: {sorted(columns)}. See docs/datasets.md."
        )
    numeric = {
        spec.target_field: source
        for spec in _NUMERIC_FIELDS
        if (source := find(spec.aliases, spec.target_field)) is not None
    }
    temperature = find(_TEMPERATURE_ALIASES, "temperature")
    if not numeric.keys() - {"age"} and temperature is None:
        raise ValueError(
            "Triagegeist file has no recognized structured triage column (vital signs or pain); "
            f"columns found: {sorted(columns)}. Extend the aliases in "
            "app/ml/data/triagegeist.py and document the correspondence."
        )
    patient_id = find(_PATIENT_ID_ALIASES, "patient identifier")
    complaint = find(_COMPLAINT_ALIASES, "chief complaint")
    identifiers = sorted(
        c for alias in _OTHER_ID_ALIASES for c in by_normalized.get(alias, []) if c not in used
    )
    return ColumnMapping(
        target=target,
        patient_id=patient_id,
        chief_complaint=complaint,
        numeric=numeric,
        temperature=temperature,
        temperature_unit=(
            _TEMPERATURE_ALIASES[normalize_column(temperature)] if temperature else None
        ),
        ignored_identifiers=identifiers,
        unused=sorted(set(columns) - used - set(identifiers)),
    )


def triagegeist_dataset_info(source_version: str) -> DatasetInfo:
    return DatasetInfo(
        name="triagegeist",
        version=f"{TRIAGEGEIST_DATASET_VERSION}+source-{source_version}",
        disclaimer=TRIAGEGEIST_DISCLAIMER,
        label_definition=TRIAGE_ACUITY_MAPPING_DESCRIPTION,
    )


def pseudonymous_group_id(patient_id: str) -> str:
    """Stable per-patient key for group splitting; pseudonymization, not anonymization."""
    return hashlib.sha256(f"triagegeist:{patient_id}".encode()).hexdigest()[:16]


def resolve_train_path(path: Path) -> Path:
    """Accept the directory containing `train.csv` or the CSV file itself."""
    candidate = path / TRAIN_FILENAME if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Triagegeist file not found: {candidate}. Download it manually from Kaggle and "
            "place train.csv in data/raw/triagegeist/ (see docs/datasets.md)."
        )
    return candidate


@dataclass
class TriagegeistReport:
    """Aggregate-only statistics; safe to log (no identifiers, no free text)."""

    rows_read: int = 0
    rows_kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    missing: Counter[str] = field(default_factory=Counter)
    implausible: Counter[str] = field(default_factory=Counter)
    temperature_from_fahrenheit: int = 0
    acuity_distribution: Counter[str] = field(default_factory=Counter)
    class_distribution: Counter[str] = field(default_factory=Counter)
    columns: ColumnMapping | None = None
    patients: set[str] = field(default_factory=set, repr=False)

    @property
    def number_of_patients(self) -> int | None:
        has_ids = self.columns is not None and self.columns.patient_id is not None
        return len(self.patients) if has_ids else None

    def as_dict(self) -> dict[str, object]:
        return {
            "rows_read": self.rows_read,
            "rows_kept": self.rows_kept,
            "rows_dropped": dict(self.dropped),
            "missing_after_cleaning": dict(self.missing),
            "implausible_set_to_missing": dict(self.implausible),
            "temperature_converted_from_fahrenheit": self.temperature_from_fahrenheit,
            "original_acuity_distribution": dict(sorted(self.acuity_distribution.items())),
            "class_distribution": dict(self.class_distribution),
            "number_of_patients": self.number_of_patients,
            "patient_level_split_possible": self.number_of_patients is not None,
            "column_mapping": self.columns.as_dict() if self.columns else None,
        }


def read_train(
    path: Path, report: TriagegeistReport, target_column: str = DEFAULT_TARGET_COLUMN
) -> Iterator[RoutingTrainingExample]:
    """Stream canonical examples from Triagegeist `train.csv`, filling `report`."""
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        columns = detect_columns(reader.fieldnames or (), target_column)
        report.columns = columns
        for row in reader:
            report.rows_read += 1
            example = _convert_row(row, columns, report)
            if example is not None:
                report.rows_kept += 1
                report.class_distribution[example.label.value] += 1
                yield example


def _convert_row(
    row: dict[str, str], columns: ColumnMapping, report: TriagegeistReport
) -> RoutingTrainingExample | None:
    acuity = parse_number(row.get(columns.target))
    label = map_triage_acuity_to_care_level(acuity)
    if label is None or acuity is None:
        report.dropped["invalid_or_missing_triage_acuity"] += 1
        return None
    group_id = None
    if columns.patient_id is not None:
        patient_id = (row.get(columns.patient_id) or "").strip()
        if not patient_id:
            report.dropped["missing_patient_id"] += 1
            return None
        group_id = pseudonymous_group_id(patient_id)
        report.patients.add(group_id)
    report.acuity_distribution[str(int(acuity))] += 1

    values: dict[str, float | None] = dict.fromkeys(STRUCTURED_FIELDS)
    for spec in _NUMERIC_FIELDS:
        if (source := columns.numeric.get(spec.target_field)) is not None:
            values[spec.target_field] = _clean_numeric(row.get(source), spec, report)
    values["temperature_celsius"] = _clean_temperature(row, columns, report)
    for name, value in values.items():
        if value is None:
            report.missing[name] += 1
    age = values.pop("age")

    complaint = None
    if columns.chief_complaint is not None:
        complaint = (row.get(columns.chief_complaint) or "").strip() or None
    return RoutingTrainingExample(
        group_id=group_id,
        label=label,
        age_range=None if age is None else AgeRange.from_age(int(age)),
        chief_complaint=complaint,
        **values,
    )


def _clean_numeric(raw: str | None, spec: _NumericField, report: TriagegeistReport) -> float | None:
    if raw is None or not raw.strip():
        return None
    value = parse_number(raw)
    if value is None or not spec.low <= value <= spec.high:
        report.implausible[spec.target_field] += 1
        return None
    return value


def _clean_temperature(
    row: dict[str, str], columns: ColumnMapping, report: TriagegeistReport
) -> float | None:
    if columns.temperature is None:
        return None
    raw = row.get(columns.temperature)
    if raw is None or not raw.strip():
        return None
    value = parse_number(raw)
    unit = columns.temperature_unit
    is_celsius = value is not None and _CELSIUS_RANGE[0] <= value <= _CELSIUS_RANGE[1]
    is_fahrenheit = value is not None and _FAHRENHEIT_RANGE[0] <= value <= _FAHRENHEIT_RANGE[1]
    if value is not None and is_celsius and unit is not TemperatureUnit.FAHRENHEIT:
        return value
    if value is not None and is_fahrenheit and unit is not TemperatureUnit.CELSIUS:
        report.temperature_from_fahrenheit += 1
        return round((value - 32) * 5 / 9, 2)
    report.implausible["temperature_celsius"] += 1
    return None
