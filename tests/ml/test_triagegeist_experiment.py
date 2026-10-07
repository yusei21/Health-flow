"""End-to-end structured Triagegeist experiment on FAKE Triagegeist-shaped data."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.core.exceptions import MLModelUnavailableError
from app.ml.classifier import RoutingClassifier
from app.ml.data.base import read_canonical_csv
from app.ml.experiments import EXPERIMENTS
from app.ml.feature_builders import FEATURE_BUILDERS, HealthFlowSymptomFeatureBuilder
from app.ml.metrics import CLASS_ORDER
from app.ml.training.evaluate import evaluate_experiment
from app.ml.training.prepare_triagegeist import prepare
from app.ml.training.train import ExperimentResult, prepare_data, run_experiment
from app.schemas.care import CareLevel
from tests.ml.fixtures.fake_triagegeist import fake_train_rows, write_fake_train

EXPERIMENT = EXPERIMENTS["structured_triagegeist_baseline"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("triagegeist")
    write_fake_train(root / "raw" / "train.csv", fake_train_rows(1500, n_patients=400, seed=3))
    prepare(root / "raw", root / "processed.csv", source_version="fake")
    return root


@pytest.fixture(scope="module")
def result(workspace: Path) -> ExperimentResult:
    return run_experiment(
        EXPERIMENT, workspace / "processed.csv", workspace / "model", workspace / "bench"
    )


def test_split_with_patient_ids_never_shares_a_patient(workspace: Path) -> None:
    dataset = read_canonical_csv(workspace / "processed.csv")
    data = prepare_data(dataset, FEATURE_BUILDERS[EXPERIMENT.feature_set], random_state=42)
    assert data.groups is not None
    assert not set(data.groups[data.split.train_index]) & set(data.groups[data.split.test_index])
    assert data.split.strategy.value.startswith("stratified_group_by_patient")


def test_split_without_patient_ids_is_row_level_and_says_so(tmp_path: Path) -> None:
    write_fake_train(tmp_path / "raw" / "train.csv", fake_train_rows(300, None, seed=4))
    prepare(tmp_path / "raw", tmp_path / "p.csv", source_version="fake")
    dataset = read_canonical_csv(tmp_path / "p.csv")
    assert dataset.number_of_patients is None
    data = prepare_data(dataset, FEATURE_BUILDERS[EXPERIMENT.feature_set], random_state=42)
    assert data.groups is None
    assert data.split.strategy.value.startswith("stratified_row")


def test_features_contain_nan_for_missing_values_not_zero(workspace: Path) -> None:
    dataset = read_canonical_csv(workspace / "processed.csv")
    data = prepare_data(dataset, FEATURE_BUILDERS[EXPERIMENT.feature_set], random_state=42)
    assert np.isnan(data.features).any()
    assert data.features.shape[1] == len(FEATURE_BUILDERS[EXPERIMENT.feature_set].feature_names)


def test_metadata_describes_dataset_split_and_features(result: ExperimentResult) -> None:
    meta = result.metadata
    assert meta.experiment == "structured_triagegeist_baseline"
    assert meta.dataset_name == "triagegeist"
    assert meta.feature_set == "triagegeist-structured-v1"
    assert "age_range_rank" in meta.features
    assert not {"group_id", "patient_id", "visit_id", "chief_complaint"} & set(meta.features)
    assert meta.split_strategy.startswith("stratified_group_by_patient")
    assert meta.number_of_patients is not None and meta.number_of_patients <= 400
    assert meta.hyperparameters["impute"]["strategy"] == "median"
    assert set(meta.metrics.cross_validation) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "neural_network_mlp",
    }


def test_writes_one_record_per_model_with_required_fields(result: ExperimentResult) -> None:
    assert len(result.benchmark_files) == 4
    for path in result.benchmark_files:
        text = path.read_text()
        assert "NaN" not in text
        record = json.loads(text)
        assert record["dataset_name"] == "triagegeist"
        assert record["git_dirty"] == record["git_commit"].endswith("-dirty")
        assert record["inference_time_seconds"] >= 0
        for key in (
            "timestamp",
            "git_commit",
            "dataset_version",
            "number_of_rows",
            "number_of_patients",
            "split_strategy",
            "feature_set",
            "random_state",
            "model_name",
            "hyperparameters",
            "training_time_seconds",
        ):
            assert key in record
        test = record["test_metrics"]
        for key in (
            "accuracy",
            "per_class",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "emergency_recall",
            "confusion_matrix",
            "under_triage_rate",
            "over_triage_rate",
            "critical_under_triage_rate",
        ):
            assert key in test


def test_triage_rates_are_consistent_with_confusion_matrix(result: ExperimentResult) -> None:
    metrics = result.metadata.metrics.held_out_test
    matrix = np.array(metrics.confusion_matrix)
    ranks = np.array([CareLevel(label).rank for label in CLASS_ORDER])
    true_rank, pred_rank = np.meshgrid(ranks, ranks, indexing="ij")
    total = matrix.sum()
    assert metrics.under_triage_rate == pytest.approx(
        matrix[pred_rank < true_rank].sum() / total, abs=1e-4
    )
    assert metrics.over_triage_rate == pytest.approx(
        matrix[pred_rank > true_rank].sum() / total, abs=1e-4
    )
    assert metrics.critical_under_triage_rate == pytest.approx(1 - metrics.emergency_recall)


def test_saved_model_loads_and_evaluation_reproduces_held_out_metrics(
    workspace: Path, result: ExperimentResult
) -> None:
    RoutingClassifier.load(workspace / "model", expected_feature_set=EXPERIMENT.feature_set)
    metrics = evaluate_experiment(EXPERIMENT, workspace / "processed.csv", workspace / "model")
    assert metrics == result.metadata.metrics.held_out_test


def test_api_refuses_triagegeist_model(workspace: Path, result: ExperimentResult) -> None:
    with pytest.raises(MLModelUnavailableError, match="feature set"):
        RoutingClassifier.load(
            workspace / "model",
            expected_feature_set=HealthFlowSymptomFeatureBuilder.feature_set,
        )
