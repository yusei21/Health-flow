"""Benchmark records for the article: one immutable JSON per (run, model)."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from app.ml.metrics import ClassificationMetrics

DEFAULT_BENCHMARK_DIR = Path("benchmarks/results")


class BenchmarkRecord(BaseModel):
    run_id: str
    timestamp: str
    git_commit: str
    git_dirty: bool
    experiment: str
    dataset_name: str
    dataset_version: str
    data_disclaimer: str
    label_definition: str
    number_of_rows: int
    number_of_patients: int | None
    train_rows: int
    test_rows: int
    feature_set: str
    features: list[str]
    split_strategy: str
    random_state: int
    model_name: str
    hyperparameters: dict[str, dict[str, object]]
    selection_criterion: str
    selected_for_deployment: bool
    cross_validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    training_time_seconds: float
    inference_time_seconds: float  # whole held-out test set
    inference_time_ms_per_row: float


def git_commit() -> str:
    """Current commit (suffixed `-dirty` with uncommitted changes), or "unknown"."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed command, no user input
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}-dirty" if status else commit


def write_record(record: BenchmarkRecord, directory: Path, started_at: datetime) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"{started_at:%Y-%m-%dT%H%M%S}_{record.experiment}_{record.model_name}_{record.run_id[:8]}"
    )
    path = directory / f"{name}.json"
    # Mode "x": an existing result is never overwritten.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)
    return path
