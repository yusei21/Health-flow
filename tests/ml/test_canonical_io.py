from pathlib import Path

from app.ml.data.base import read_canonical_csv, write_canonical_csv
from app.ml.data.schemas import DatasetInfo, RoutingTrainingExample
from app.schemas.care import CareLevel
from app.schemas.symptoms import Symptom

INFO = DatasetInfo(name="t", version="v", disclaimer="FAKE", label_definition="test")


def test_round_trip_keeps_none_and_empty_lists_distinct(tmp_path: Path) -> None:
    examples = [
        RoutingTrainingExample(label=CareLevel.EMERGENCY, group_id="a", heart_rate=120.5),
        RoutingTrainingExample(label=CareLevel.PRIMARY_CARE, symptoms=[], pain=0),
        RoutingTrainingExample(label=CareLevel.URGENT_CARE, symptoms=[Symptom.FEVER]),
    ]
    path = tmp_path / "d.csv"
    assert write_canonical_csv(path, INFO, examples) == 3
    dataset = read_canonical_csv(path)
    assert dataset.examples == examples
    assert dataset.info == INFO
    assert dataset.examples[0].symptoms is None and dataset.examples[1].symptoms == []


def test_number_of_patients_is_none_without_group_ids(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    write_canonical_csv(path, INFO, [RoutingTrainingExample(label=CareLevel.EMERGENCY)])
    assert read_canonical_csv(path).number_of_patients is None
