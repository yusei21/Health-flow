"""End-to-end structured MIMIC experiment on FAKE MIMIC-shaped data."""

import json
from pathlib import Path

import pytest

from app.core.exceptions import MLModelUnavailableError
from app.ml.classifier import RoutingClassifier
from app.ml.experiments import EXPERIMENTS
from app.ml.feature_builders import HealthFlowSymptomFeatureBuilder
from app.ml.training.evaluate import evaluate_experiment
from app.ml.training.prepare_mimic import prepare
from app.ml.training.train import ExperimentResult, run_experiment
from tests.ml.fixtures.fake_mimic import fake_triage_rows, write_fake_triage

EXPERIMENT = EXPERIMENTS["structured_mimic_baseline"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("mimic")
    write_fake_triage(root / "raw" / "triage.csv.gz", fake_triage_rows(1500, 400, seed=3))
    prepare(root / "raw", root / "processed.csv", source_version="fake")
    return root


@pytest.fixture(scope="module")
def result(workspace: Path) -> ExperimentResult:
    return run_experiment(
        EXPERIMENT, workspace / "processed.csv", workspace / "model", workspace / "bench"
    )


def test_metadata_describes_dataset_split_and_features(result: ExperimentResult) -> None:
    meta = result.metadata
    assert meta.experiment == "structured_mimic_baseline"
    assert meta.dataset_name == "mimic-iv-ed"
    assert meta.feature_set == "mimic-structured-vitals-v1"
    assert meta.split_strategy.startswith("stratified_group_by_patient")
    assert meta.number_of_patients is not None and meta.number_of_patients <= 400
    assert meta.hyperparameters["impute"]["strategy"] == "median"
    assert set(meta.metrics.cross_validation) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "neural_network_mlp",
    }


def test_writes_one_valid_json_record_per_model(result: ExperimentResult) -> None:
    assert len(result.benchmark_files) == 4
    for path in result.benchmark_files:
        record = json.loads(path.read_text())  # strict JSON: NaN would fail elsewhere
        assert "NaN" not in path.read_text()
        assert record["number_of_patients"] == result.metadata.number_of_patients
        assert record["selected_for_deployment"] == (record["model_name"] == result.selected_model)
        for key in (
            "timestamp",
            "git_commit",
            "training_time_seconds",
            "inference_time_ms_per_row",
            "hyperparameters",
            "split_strategy",
        ):
            assert key in record


def test_rerun_never_overwrites_previous_records(workspace: Path, result: ExperimentResult) -> None:
    again = run_experiment(
        EXPERIMENT, workspace / "processed.csv", workspace / "model2", workspace / "bench"
    )
    assert not set(again.benchmark_files) & set(result.benchmark_files)
    assert len(list((workspace / "bench").glob("*.json"))) == 8


def test_saved_model_loads_and_evaluation_reproduces_held_out_metrics(
    workspace: Path, result: ExperimentResult
) -> None:
    RoutingClassifier.load(workspace / "model", expected_feature_set=EXPERIMENT.feature_set)
    metrics = evaluate_experiment(EXPERIMENT, workspace / "processed.csv", workspace / "model")
    assert metrics == result.metadata.metrics.held_out_test


def test_api_refuses_vitals_model(workspace: Path, result: ExperimentResult) -> None:
    with pytest.raises(MLModelUnavailableError, match="feature set"):
        RoutingClassifier.load(
            workspace / "model",
            expected_feature_set=HealthFlowSymptomFeatureBuilder.feature_set,
        )


def test_evaluation_refuses_a_changed_dataset(workspace: Path, result: ExperimentResult) -> None:
    write_fake_triage(workspace / "raw2" / "triage.csv", fake_triage_rows(900, 300, seed=9))
    prepare(workspace / "raw2", workspace / "other.csv", source_version="fake")
    with pytest.raises(ValueError, match="retrain"):
        evaluate_experiment(EXPERIMENT, workspace / "other.csv", workspace / "model")
