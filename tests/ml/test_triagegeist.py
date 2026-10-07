import json
import math
from pathlib import Path

import pytest

from app.ml.data.base import read_canonical_csv
from app.ml.data.schemas import RoutingTrainingExample
from app.ml.data.triagegeist import (
    TemperatureUnit,
    TriagegeistReport,
    detect_columns,
    map_triage_acuity_to_care_level,
    pseudonymous_group_id,
    read_train,
    resolve_train_path,
)
from app.ml.training.prepare_triagegeist import prepare
from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange
from tests.ml.fixtures.fake_triagegeist import fake_train_rows, write_fake_train

ROW = {
    "patient_id": "P000001",
    "visit_id": "V0000001",
    "age": "45",
    "heart_rate": "88",
    "respiratory_rate": "18",
    "spo2": "97",
    "systolic_bp": "130",
    "diastolic_bp": "80",
    "temperature": "37.0",
    "pain_score": "5",
    "arrival_mode": "walk-in",
    "chief_complaint": "FAKE headache",
    "triage_acuity": "3",
}


def read(path: Path) -> tuple[list[RoutingTrainingExample], TriagegeistReport]:
    report = TriagegeistReport()
    return list(read_train(path, report)), report


@pytest.mark.parametrize(
    ("acuity", "expected"),
    [
        (1, CareLevel.EMERGENCY),
        (2, CareLevel.EMERGENCY),
        (3, CareLevel.URGENT_CARE),
        (4, CareLevel.PRIMARY_CARE),
        (5, CareLevel.PRIMARY_CARE),
        ("2", CareLevel.EMERGENCY),
        ("4.0", CareLevel.PRIMARY_CARE),
    ],
)
def test_acuity_mapping_for_every_valid_value(acuity: object, expected: CareLevel) -> None:
    assert map_triage_acuity_to_care_level(acuity) is expected


@pytest.mark.parametrize("acuity", [None, "", 0, 6, -1, 2.5, "abc", math.nan, True])
def test_invalid_acuity_maps_to_none_never_to_a_class(acuity: object) -> None:
    assert map_triage_acuity_to_care_level(acuity) is None


def test_detects_columns_by_alias_and_keeps_ids_and_unknowns_out() -> None:
    columns = detect_columns(ROW)
    assert columns.target == "triage_acuity"
    assert columns.patient_id == "patient_id"
    assert columns.numeric["oxygen_saturation"] == "spo2"
    assert columns.numeric["pain"] == "pain_score"
    assert columns.temperature == "temperature"
    assert columns.temperature_unit is TemperatureUnit.PER_VALUE_RANGE
    assert columns.ignored_identifiers == ["visit_id"]
    assert columns.unused == ["arrival_mode"]


def test_detection_normalizes_header_spelling() -> None:
    columns = detect_columns(["Triage Acuity", "Heart Rate", "SpO2", "Temp (F)", "Patient-ID"])
    assert columns.target == "Triage Acuity"
    assert columns.numeric == {"heart_rate": "Heart Rate", "oxygen_saturation": "SpO2"}
    assert columns.temperature_unit is TemperatureUnit.FAHRENHEIT
    assert columns.patient_id == "Patient-ID"


def test_missing_target_fails_with_clear_message() -> None:
    with pytest.raises(ValueError, match=r"triage_acuity.*columns found"):
        detect_columns(["heart_rate", "spo2"])


def test_file_without_structured_features_fails() -> None:
    with pytest.raises(ValueError, match="no recognized structured"):
        detect_columns(["triage_acuity", "age", "chief_complaint"])


def test_two_columns_for_the_same_field_fail_instead_of_silently_picking() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        detect_columns(["triage_acuity", "heart_rate", "pulse"])


def test_converts_row_without_identifiers_or_free_text_features(tmp_path: Path) -> None:
    [example], report = read(write_fake_train(tmp_path / "train.csv", [ROW]))
    assert example.label is CareLevel.URGENT_CARE
    assert example.group_id == pseudonymous_group_id("P000001") != "P000001"
    assert (example.heart_rate, example.oxygen_saturation, example.pain) == (88, 97, 5)
    assert example.temperature_celsius == 37.0
    assert example.age_range is AgeRange.ADULT
    assert example.chief_complaint == "FAKE headache"
    assert report.rows_kept == 1 and report.number_of_patients == 1
    assert report.acuity_distribution == {"3": 1}


@pytest.mark.parametrize(
    ("column", "raw", "expected", "converted"),
    [
        ("temperature", "98.6", 37.0, 1),  # ambiguous name, value in °F range
        ("temperature", "37.0", 37.0, 0),  # ambiguous name, value in °C range
        ("temperature_f", "98.6", 37.0, 1),
        ("temperature_c", "38.5", 38.5, 0),
    ],
)
def test_temperature_units(
    tmp_path: Path, column: str, raw: str, expected: float, converted: int
) -> None:
    row = {k: v for k, v in ROW.items() if k != "temperature"} | {column: raw}
    [example], report = read(write_fake_train(tmp_path / "train.csv", [row]))
    assert example.temperature_celsius == pytest.approx(expected)
    assert report.temperature_from_fahrenheit == converted


@pytest.mark.parametrize(("column", "raw"), [("temperature", "60"), ("temperature_c", "98.6")])
def test_temperature_outside_the_unit_range_becomes_missing(
    tmp_path: Path, column: str, raw: str
) -> None:
    row = {k: v for k, v in ROW.items() if k != "temperature"} | {column: raw}
    [example], report = read(write_fake_train(tmp_path / "train.csv", [row]))
    assert example.temperature_celsius is None
    assert report.implausible["temperature_celsius"] == 1


def test_missing_and_implausible_values_become_none_not_zero(tmp_path: Path) -> None:
    row = {**ROW, "heart_rate": "", "spo2": "250", "pain_score": "n/a", "age": ""}
    [example], report = read(write_fake_train(tmp_path / "train.csv", [row]))
    assert example.heart_rate is None and example.oxygen_saturation is None
    assert example.pain is None and example.age_range is None
    assert report.implausible == {"oxygen_saturation": 1, "pain": 1}
    assert report.missing["heart_rate"] == 1 and report.missing["age"] == 1


def test_pain_zero_is_a_real_value(tmp_path: Path) -> None:
    [example], _ = read(write_fake_train(tmp_path / "train.csv", [{**ROW, "pain_score": "0"}]))
    assert example.pain == 0


def test_invalid_acuity_and_missing_patient_rows_are_dropped_and_counted(tmp_path: Path) -> None:
    rows = [
        ROW,
        {**ROW, "triage_acuity": ""},
        {**ROW, "triage_acuity": "7"},
        {**ROW, "triage_acuity": "2.5"},
        {**ROW, "patient_id": " "},
    ]
    examples, report = read(write_fake_train(tmp_path / "train.csv", rows))
    assert len(examples) == 1
    assert report.dropped == {"invalid_or_missing_triage_acuity": 3, "missing_patient_id": 1}


def test_without_patient_column_there_is_no_group_and_no_patient_count(tmp_path: Path) -> None:
    rows = fake_train_rows(50, n_patients=None, seed=1, invalid_rate=0)
    examples, report = read(write_fake_train(tmp_path / "train.csv", rows))
    assert len(examples) == 50
    assert all(ex.group_id is None for ex in examples)
    assert report.number_of_patients is None
    assert report.as_dict()["patient_level_split_possible"] is False


def test_absent_optional_field_is_reported_as_unmapped(tmp_path: Path) -> None:
    row = {k: v for k, v in ROW.items() if k not in {"spo2", "age"}}
    [example], report = read(write_fake_train(tmp_path / "train.csv", [row]))
    assert example.oxygen_saturation is None and example.age_range is None
    mapping = report.as_dict()["column_mapping"]
    assert isinstance(mapping, dict)
    assert mapping["canonical_fields"]["oxygen_saturation"] is None


def test_resolve_accepts_directory_or_file_and_explains_missing_file(tmp_path: Path) -> None:
    csv_file = write_fake_train(tmp_path / "train.csv", [ROW])
    assert resolve_train_path(tmp_path) == csv_file
    assert resolve_train_path(csv_file) == csv_file
    with pytest.raises(FileNotFoundError, match=r"manually from Kaggle.*datasets\.md"):
        resolve_train_path(tmp_path / "missing")


def test_prepare_writes_processed_dataset_and_required_metadata(tmp_path: Path) -> None:
    rows = fake_train_rows(400, n_patients=120, seed=2)
    source = write_fake_train(tmp_path / "raw" / "train.csv", rows)
    output = tmp_path / "processed" / "routing_triagegeist_v1.csv"
    report = prepare(source.parent, output, source_version="test")

    dataset = read_canonical_csv(output)
    assert len(dataset.examples) == report.rows_kept
    assert dataset.number_of_patients == report.number_of_patients
    assert dataset.info.name == "triagegeist"

    meta = json.loads(output.with_suffix(".meta.json").read_text())
    for key in (
        "dataset_name",
        "dataset_version",
        "source",
        "number_of_rows",
        "class_distribution",
        "discarded_rows",
        "mapping_used",
        "feature_set",
        "disclaimer",
    ):
        assert key in meta
    assert meta["number_of_rows"] == report.rows_kept
    assert meta["discarded_rows"]["total"] == report.rows_read - report.rows_kept > 0
    assert meta["mapping_used"]["table"]["4"] == "PRIMARY_CARE"
    assert meta["feature_set"] == "triagegeist-structured-v1"
    assert "NOT an SUS protocol" in meta["disclaimer"]
    assert sum(meta["class_distribution"].values()) == report.rows_kept

    content = output.read_text()
    assert all(r["patient_id"] not in content and r["visit_id"] not in content for r in rows)
    assert "arrival_mode" not in content
