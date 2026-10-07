import math
from pathlib import Path

import pytest

from app.ml.data.base import read_canonical_csv
from app.ml.data.mimic_ed import (
    PreparationReport,
    map_esi_to_care_level,
    pseudonymous_group_id,
    read_triage,
    resolve_triage_path,
)
from app.ml.data.schemas import RoutingTrainingExample
from app.ml.training.prepare_mimic import prepare
from app.schemas.care import CareLevel
from tests.ml.fixtures.fake_mimic import fake_triage_rows, write_fake_triage

ROW = {
    "subject_id": "10000001",
    "stay_id": "30000001",
    "temperature": "98.6",
    "heartrate": "88",
    "resprate": "18",
    "o2sat": "97",
    "sbp": "130",
    "dbp": "80",
    "pain": "5",
    "acuity": "3.0",
    "chiefcomplaint": "FAKE headache",
}


def read(path: Path) -> tuple[list[RoutingTrainingExample], PreparationReport]:
    report = PreparationReport()
    return list(read_triage(path, report)), report


@pytest.mark.parametrize(
    ("acuity", "expected"),
    [
        (1, CareLevel.EMERGENCY),
        (2, CareLevel.EMERGENCY),
        (3, CareLevel.URGENT_CARE),
        (4, CareLevel.PRIMARY_CARE),
        (5, CareLevel.PRIMARY_CARE),
        ("1", CareLevel.EMERGENCY),
        ("3.0", CareLevel.URGENT_CARE),
        (5.0, CareLevel.PRIMARY_CARE),
    ],
)
def test_esi_mapping_for_every_valid_value(acuity: object, expected: CareLevel) -> None:
    assert map_esi_to_care_level(acuity) is expected


@pytest.mark.parametrize(
    "acuity", [None, "", "  ", 0, 6, -1, 2.5, "2.5", "abc", "nan", math.nan, math.inf, True, []]
)
def test_invalid_acuity_maps_to_none_never_to_a_class(acuity: object) -> None:
    assert map_esi_to_care_level(acuity) is None


@pytest.mark.parametrize("filename", ["triage.csv", "triage.csv.gz"])
def test_parses_csv_and_csv_gz_identically(tmp_path: Path, filename: str) -> None:
    rows = fake_triage_rows(200, 50, seed=1)
    examples, _ = read(write_fake_triage(tmp_path / filename, rows))
    reference, _ = read(write_fake_triage(tmp_path / "ref" / "triage.csv", rows))
    assert examples == reference
    assert len(examples) > 150


def test_converts_units_and_keeps_only_routing_fields(tmp_path: Path) -> None:
    [example], report = read(write_fake_triage(tmp_path / "triage.csv", [ROW]))
    assert example.label is CareLevel.URGENT_CARE
    assert example.temperature_celsius == pytest.approx(37.0)
    assert (example.heart_rate, example.oxygen_saturation, example.pain) == (88, 97, 5)
    assert example.group_id == pseudonymous_group_id("10000001") != "10000001"
    assert example.symptoms is None and example.age_range is None  # not in triage table
    assert report.rows_kept == 1


def test_missing_implausible_and_text_values_become_none_and_are_counted(tmp_path: Path) -> None:
    row = {**ROW, "heartrate": "", "o2sat": "250", "temperature": "37.2", "pain": "unable"}
    [example], report = read(write_fake_triage(tmp_path / "triage.csv", [row]))
    assert example.heart_rate is None
    assert example.oxygen_saturation is None  # implausible
    assert example.temperature_celsius is None  # 37.2 °F is implausible; no unit guessing
    assert example.pain is None
    assert report.implausible == {"oxygen_saturation": 1, "temperature_celsius": 1}
    assert report.pain_invalid == 1
    assert report.missing["heart_rate"] == 1


def test_invalid_acuity_and_missing_subject_rows_are_dropped(tmp_path: Path) -> None:
    rows = [ROW, {**ROW, "acuity": ""}, {**ROW, "acuity": "9"}, {**ROW, "subject_id": ""}]
    examples, report = read(write_fake_triage(tmp_path / "triage.csv", rows))
    assert len(examples) == 1
    assert report.dropped == {"invalid_or_missing_acuity": 2, "missing_subject_id": 1}


def test_absent_optional_columns_are_reported_not_invented(tmp_path: Path) -> None:
    row = {k: v for k, v in ROW.items() if k not in {"pain", "o2sat"}}
    [example], report = read(write_fake_triage(tmp_path / "triage.csv", [row]))
    assert report.absent_columns == ["o2sat", "pain"]
    assert example.pain is None and example.oxygen_saturation is None


def test_missing_required_column_fails_loudly(tmp_path: Path) -> None:
    row = {k: v for k, v in ROW.items() if k != "acuity"}
    with pytest.raises(ValueError, match="acuity"):
        read(write_fake_triage(tmp_path / "triage.csv", [row]))


def test_resolve_accepts_directory_and_falls_back_between_csv_and_gz(tmp_path: Path) -> None:
    csv_file = write_fake_triage(tmp_path / "triage.csv", [ROW])
    assert resolve_triage_path(tmp_path) == csv_file
    assert resolve_triage_path(tmp_path / "triage.csv.gz") == csv_file
    with pytest.raises(FileNotFoundError, match=r"datasets\.md"):
        resolve_triage_path(tmp_path / "nothing")


def test_prepare_writes_canonical_dataset_without_raw_identifiers(tmp_path: Path) -> None:
    rows = fake_triage_rows(300, 80, seed=2)
    source = write_fake_triage(tmp_path / "raw" / "triage.csv.gz", rows)
    output = tmp_path / "processed" / "routing_mimic_v1.csv"
    report = prepare(source.parent, output, source_version="test")

    dataset = read_canonical_csv(output)
    assert len(dataset.examples) == report.rows_kept
    assert dataset.number_of_patients == len(report.patients)
    content = output.read_text()
    assert all(row["subject_id"] not in content and row["stay_id"] not in content for row in rows)
    assert "ESI 1-2" in dataset.info.label_definition
