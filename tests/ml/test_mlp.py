"""neural_network_mlp candidate across all registered experiments (FAKE/synthetic data only)."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from app.ml.classifier import MODEL_FILENAME, RoutingClassifier
from app.ml.data.synthetic import generate_examples, save_dataset, to_training_example
from app.ml.experiments import EXPERIMENTS
from app.ml.feature_builders import HealthFlowSymptomFeatureBuilder
from app.ml.metrics import CLASS_ORDER
from app.ml.training.evaluate import evaluate_experiment
from app.ml.training.prepare_mimic import prepare as prepare_mimic
from app.ml.training.prepare_triagegeist import prepare as prepare_triagegeist
from app.ml.training.train import (
    _record_convergence_warnings,
    candidate_models,
    run_experiment,
    training_diagnostics,
)
from benchmarks.scripts.summarize_results import summarize
from tests.ml.fixtures.fake_mimic import fake_triage_rows, write_fake_triage
from tests.ml.fixtures.fake_triagegeist import fake_train_rows, write_fake_train

MLP = "neural_network_mlp"


def _synthetic(root: Path) -> Path:
    save_dataset(generate_examples(600, seed=5), root / "synthetic.csv")
    return root / "synthetic.csv"


def _mimic(root: Path) -> Path:
    write_fake_triage(root / "raw" / "triage.csv", fake_triage_rows(600, 200, seed=5))
    prepare_mimic(root / "raw", root / "mimic.csv", source_version="fake")
    return root / "mimic.csv"


def _triagegeist(root: Path) -> Path:
    write_fake_train(root / "raw" / "train.csv", fake_train_rows(600, n_patients=200, seed=5))
    prepare_triagegeist(root / "raw", root / "triagegeist.csv", source_version="fake")
    return root / "triagegeist.csv"


DATASETS = {
    "synthetic_baseline": _synthetic,
    "structured_mimic_baseline": _mimic,
    "structured_triagegeist_baseline": _triagegeist,
}


def test_mlp_candidate_has_the_specified_architecture() -> None:
    pipeline = candidate_models(random_state=42)[MLP]
    assert [name for name, _ in pipeline.steps] == ["impute", "scale", "model"]
    mlp = pipeline.named_steps["model"]
    assert isinstance(mlp, MLPClassifier)
    assert (
        mlp.get_params()
        | {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": 42,
        }
        == mlp.get_params()
    )


def test_existing_candidates_are_preserved() -> None:
    assert list(candidate_models()) == [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        MLP,
    ]


@pytest.mark.parametrize("experiment_name", sorted(DATASETS))
def test_mlp_trains_saves_loads_and_predicts_in_every_experiment(
    tmp_path: Path, experiment_name: str
) -> None:
    experiment = EXPERIMENTS[experiment_name]
    dataset = DATASETS[experiment_name](tmp_path)
    result = run_experiment(
        experiment, dataset, tmp_path / "model", tmp_path / "bench", models=[MLP]
    )

    meta = result.metadata
    assert result.selected_model == meta.model_type == MLP
    model_params = meta.hyperparameters["model"]
    assert model_params["hidden_layer_sizes"] == [64, 32]
    assert {k: model_params[k] for k in ("activation", "solver", "max_iter")} == {
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
    }
    assert model_params["early_stopping"] is True
    assert model_params["validation_fraction"] == 0.1
    assert model_params["n_iter_no_change"] == 20
    assert model_params["random_state"] == meta.random_state == 42
    diagnostics = meta.training_diagnostics
    assert isinstance(diagnostics["n_iter"], int) and 0 < diagnostics["n_iter"] <= 500
    assert isinstance(diagnostics["converged"], bool)
    assert "stopped_early" in diagnostics and "convergence_warnings" in diagnostics

    classifier = RoutingClassifier.load(
        tmp_path / "model", expected_feature_set=experiment.feature_set
    )
    assert sorted(classifier.classes) == sorted(CLASS_ORDER)
    data_rows = np.zeros((3, len(meta.features)))
    probabilities = classifier.predict_proba(data_rows)
    assert probabilities.shape == (3, len(CLASS_ORDER))
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert set(classifier.predict_labels(data_rows)) <= set(CLASS_ORDER)

    pipeline = joblib.load(tmp_path / "model" / MODEL_FILENAME)  # our own artifact
    assert isinstance(pipeline, Pipeline)
    assert set(pipeline.predict(data_rows)) <= set(CLASS_ORDER)

    metrics = evaluate_experiment(experiment, dataset, tmp_path / "model")
    assert metrics == meta.metrics.held_out_test

    [record_file] = result.benchmark_files
    record = json.loads(record_file.read_text())
    assert record["model_name"] == MLP
    assert record["training_diagnostics"] == diagnostics
    assert record["inference_time_seconds"] >= 0 and record["training_time_seconds"] > 0
    assert "NaN" not in record_file.read_text()


def test_all_candidates_share_split_and_protocol(tmp_path: Path) -> None:
    result = run_experiment(
        EXPERIMENTS["structured_triagegeist_baseline"],
        _triagegeist(tmp_path),
        tmp_path / "model",
        tmp_path / "bench",
    )
    records = [json.loads(path.read_text()) for path in result.benchmark_files]
    assert {r["model_name"] for r in records} >= {MLP}
    for key in ("split_strategy", "train_rows", "test_rows", "features", "selection_criterion"):
        assert len({json.dumps(r[key]) for r in records}) == 1, key
    assert sum(r["selected_for_deployment"] for r in records) == 1


def test_non_convergence_is_recorded_not_hidden() -> None:
    builder = HealthFlowSymptomFeatureBuilder()
    examples = [to_training_example(ex) for ex in generate_examples(300, seed=1)]
    features = np.array([builder.build(ex) for ex in examples])
    labels = np.array([ex.label.value for ex in examples])
    # Deliberately starved of iterations to trigger sklearn's ConvergenceWarning.
    model = clone(candidate_models()[MLP]).set_params(
        model__max_iter=3, model__early_stopping=False
    )
    with _record_convergence_warnings() as fit_warnings:
        fitted = model.fit(features, labels)
    diagnostics = training_diagnostics(fitted, fit_warnings, cv_warnings=[])
    assert diagnostics["converged"] is False
    assert diagnostics["n_iter"] == 3
    convergence_warnings = diagnostics["convergence_warnings"]
    assert isinstance(convergence_warnings, list)
    assert any("Maximum iterations" in message for message in convergence_warnings)
    assert "stopped_early" not in diagnostics  # early stopping disabled here


def test_unknown_model_selection_fails() -> None:
    with pytest.raises(ValueError, match="no candidate model named"):
        run_experiment(
            EXPERIMENTS["synthetic_baseline"],
            Path("unused.csv"),
            Path("unused"),
            models=["nope"],
        )


def test_other_warnings_pass_through_the_recorder() -> None:
    with (
        pytest.warns(UserWarning, match="unrelated"),
        _record_convergence_warnings() as messages,
    ):
        warnings.warn("unrelated", UserWarning, stacklevel=1)
    assert messages == []


def test_summary_lists_mlp_and_tolerates_old_records(tmp_path: Path) -> None:
    run_experiment(
        EXPERIMENTS["synthetic_baseline"],
        _synthetic(tmp_path),
        tmp_path / "model",
        tmp_path / "bench",
        models=[MLP],
    )
    [new_record] = tmp_path.glob("bench/*.json")
    old = json.loads(new_record.read_text())
    del old["training_diagnostics"]
    (tmp_path / "bench" / "0000_old.json").write_text(json.dumps(old))
    table = summarize(tmp_path / "bench")
    assert MLP in table
    assert "| n/a |" in table
