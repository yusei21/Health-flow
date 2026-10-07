"""Canonical processed-dataset CSV format (+ `.meta.json` sidecar with DatasetInfo)."""

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from app.ml.data.schemas import DatasetInfo, RoutingTrainingExample, TrainingDataset

CANONICAL_COLUMNS: tuple[str, ...] = tuple(RoutingTrainingExample.model_fields)
_LIST_FIELDS = frozenset({"symptoms", "risk_factors"})


def meta_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def write_canonical_csv(
    path: Path,
    info: DatasetInfo,
    examples: Iterable[RoutingTrainingExample],
    extra_meta: dict[str, object] | None = None,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CANONICAL_COLUMNS)
        for example in examples:
            writer.writerow(_to_cells(example))
            rows += 1
    meta = {**info.model_dump(), "rows": rows, **(extra_meta or {})}
    meta_path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def read_canonical_csv(path: Path) -> TrainingDataset:
    info = DatasetInfo.model_validate_json(meta_path(path).read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANONICAL_COLUMNS:
            raise ValueError(f"canonical dataset columns must be {CANONICAL_COLUMNS}")
        examples = [_from_cells(row) for row in reader]
    return TrainingDataset(info=info, examples=examples)


def _to_cells(example: RoutingTrainingExample) -> list[str]:
    data = example.model_dump(mode="json")
    # Lists are JSON so that "not available" ("") and "empty" ("[]") stay distinct.
    return [
        ""
        if data[col] is None
        else json.dumps(data[col])
        if col in _LIST_FIELDS
        else str(data[col])
        for col in CANONICAL_COLUMNS
    ]


def _from_cells(row: dict[str, str]) -> RoutingTrainingExample:
    values: dict[str, object] = {}
    for column, cell in row.items():
        if cell == "":
            values[column] = None
        elif column in _LIST_FIELDS:
            values[column] = json.loads(cell)
        else:
            values[column] = cell
    return RoutingTrainingExample.model_validate(values)
