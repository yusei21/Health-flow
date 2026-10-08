"""Train, compare and persist routing classifiers for a registered experiment.

Never runs inside the API. Usage:
    python -m app.ml.training.train --experiment synthetic_baseline
    python -m app.ml.training.train --experiment structured_mimic_baseline
    python -m app.ml.training.train --experiment structured_triagegeist_baseline

Protocol: split train/test (grouped by patient when available) → cross-validate each
candidate on the training part only → select by CV → fit on the training part →
evaluate every candidate once on the held-out test for reporting. The test set never
influences selection, early stopping or hyperparameters (the MLP's early-stopping
validation split is carved out of whatever data `fit` receives: a CV training fold or
the training part, never the held-out test).
"""

import argparse
import logging
import math
import time
import uuid
import warnings
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.classifier import MODEL_FILENAME, ModelMetadata, file_sha256, write_metadata
from app.ml.data.schemas import TrainingDataset
from app.ml.experiments import EXPERIMENTS, Experiment
from app.ml.feature_builders import FEATURE_BUILDERS, FeatureBuilder
from app.ml.metrics import CLASS_ORDER, ClassificationMetrics, TrainingMetrics, compute_metrics
from app.ml.splits import DataSplit, cv_folds, split_train_test
from app.ml.training.benchmark import (
    DEFAULT_BENCHMARK_DIR,
    BenchmarkRecord,
    git_commit,
    write_record,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_VERSION = "0.2.0"
SELECTION_CRITERION = "max(round(cv_emergency_recall, 2), cv_macro_f1)"


def _imputer() -> SimpleImputer:
    # Median imputation + missingness indicators, fitted on training folds only.
    # A no-op for feature sets without missing values.
    return SimpleImputer(strategy="median", add_indicator=True)


def candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    # class_weight="balanced": EMERGENCY is a minority class and its recall matters most.
    return {
        "logistic_regression": Pipeline(
            [
                ("impute", _imputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=random_state
                    ),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("impute", _imputer()),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", _imputer()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        # Feed-forward neural baseline for tabular data. MLPClassifier has no
        # class_weight, so unlike the others it is NOT class-balanced (documented).
        "neural_network_mlp": Pipeline(
            [
                ("impute", _imputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def hyperparameters(pipeline: Pipeline) -> dict[str, dict[str, object]]:
    return {
        name: {key: _json_safe(value) for key, value in step.get_params(deep=False).items()}
        for name, step in pipeline.steps
    }


def _json_safe(value: object) -> object:
    # NaN/inf (e.g. SimpleImputer.missing_values) would produce invalid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value if value is None or isinstance(value, bool | int | float | str) else repr(value)


@contextmanager
def _record_convergence_warnings() -> Iterator[list[str]]:
    """Collect ConvergenceWarning messages so they are recorded, not hidden.

    Any other warning is re-emitted unchanged.
    """
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield messages
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            messages.append(str(warning.message))
        else:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )


def training_diagnostics(
    model: Pipeline, fit_warnings: list[str], cv_warnings: list[str]
) -> dict[str, object]:
    """Iterations and convergence of the final fit (iterative estimators only)."""
    estimator = model.named_steps["model"]
    diagnostics: dict[str, object] = {
        "converged": not fit_warnings,
        "convergence_warnings": sorted(set(fit_warnings)),
        "cv_convergence_warning_count": len(cv_warnings),
    }
    if (n_iter := getattr(estimator, "n_iter_", None)) is not None:
        diagnostics["n_iter"] = int(np.max(n_iter))
    if isinstance(estimator, MLPClassifier) and estimator.early_stopping:
        diagnostics["stopped_early"] = estimator.n_iter_ < estimator.max_iter
        # Accuracy on the internal validation split carved from the training data.
        diagnostics["best_internal_validation_score"] = round(
            float(estimator.best_validation_score_), 4
        )
    return diagnostics


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model: Pipeline
    cv_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    training_time_seconds: float
    inference_time_seconds: float
    inference_time_ms_per_row: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    selected_model: str
    candidates: dict[str, CandidateResult]
    metadata: ModelMetadata
    benchmark_files: list[Path]


@dataclass(frozen=True)
class PreparedData:
    features: NDArray[np.float64]
    labels: NDArray[np.str_]
    groups: NDArray[np.str_] | None
    split: DataSplit


def prepare_data(
    dataset: TrainingDataset, builder: FeatureBuilder, random_state: int
) -> PreparedData:
    """Features, labels and the deterministic train/test split (shared with evaluate)."""
    features = np.array([builder.build(ex) for ex in dataset.examples], dtype=np.float64)
    labels = np.array([ex.label.value for ex in dataset.examples])
    groups = (
        np.array([ex.group_id for ex in dataset.examples])
        if dataset.number_of_patients is not None
        else None
    )
    return PreparedData(features, labels, groups, split_train_test(labels, groups, random_state))


def _selection_key(metrics: ClassificationMetrics) -> tuple[float, float]:
    return (round(metrics.emergency_recall, 2), metrics.macro_f1)


def _evaluate_candidate(
    name: str, model: Pipeline, data: PreparedData, random_state: int
) -> CandidateResult:
    train_idx, test_idx = data.split.train_index, data.split.test_index
    x_train, y_train = data.features[train_idx], data.labels[train_idx]
    groups_train = data.groups[train_idx] if data.groups is not None else None
    folds = list(cv_folds(y_train, groups_train, random_state))
    with _record_convergence_warnings() as cv_warnings:
        cv_pred = cross_val_predict(clone(model), x_train, y_train, cv=folds)

    with _record_convergence_warnings() as fit_warnings:
        started = time.perf_counter()
        fitted = clone(model).fit(x_train, y_train)
        training_time = time.perf_counter() - started
    diagnostics = training_diagnostics(fitted, fit_warnings, cv_warnings)
    if fit_warnings or cv_warnings:
        logger.warning(
            "%s did not converge in %d of %d fits (ConvergenceWarning recorded in benchmark)",
            name,
            len(fit_warnings) + len(cv_warnings),
            len(folds) + 1,
        )
    started = time.perf_counter()
    test_pred = fitted.predict(data.features[test_idx])
    inference_time = time.perf_counter() - started

    return CandidateResult(
        name=name,
        model=fitted,
        cv_metrics=compute_metrics(y_train, cv_pred),
        test_metrics=compute_metrics(data.labels[test_idx], test_pred),
        training_time_seconds=round(training_time, 3),
        inference_time_seconds=round(inference_time, 4),
        inference_time_ms_per_row=round(inference_time * 1000 / max(len(test_idx), 1), 5),
        diagnostics=diagnostics,
    )


CV_MIN_SUPPORT = 2


def validate_external_evaluation_split(data: PreparedData, random_state: int) -> None:
    """Refuse three-class evaluation when a held-out class or CV fold is absent.

    Group-preserving splits can be badly imbalanced even when all labels exist
    in the full dataset. Never present undefined per-class recall as zero.
    """
    train_idx, test_idx = data.split.train_index, data.split.test_index
    train_labels, test_labels = data.labels[train_idx], data.labels[test_idx]
    groups_train = data.groups[train_idx] if data.groups is not None else None
    partitions = [("train", train_labels), ("held-out test", test_labels)]
    for name, labels in partitions:
        counts = {label: int(np.count_nonzero(labels == label)) for label in CLASS_ORDER}
        if min(counts.values()) < CV_MIN_SUPPORT:
            raise ValueError(
                f"Insufficient support in {name}: {counts}. "
                "External three-class benchmarking needs more examples for each class."
            )
    folds = list(cv_folds(train_labels, groups_train, random_state))
    partitions = [
        (f"CV fold {i} validation", train_labels[valid])
        for i, (_, valid) in enumerate(folds, start=1)
    ]
    for name, labels in partitions:
        counts = {label: int(np.count_nonzero(labels == label)) for label in CLASS_ORDER}
        if min(counts.values()) < CV_MIN_SUPPORT:
            raise ValueError(
                f"Insufficient support in {name}: {counts}. "
                "Each class needs at least two independent examples per evaluation "
                "partition; use a larger dataset, not duplicated or invented rows."
            )


def run_experiment(
    experiment: Experiment,
    dataset_path: Path,
    model_dir: Path,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    random_state: int = RANDOM_STATE,
    models: Collection[str] | None = None,
) -> ExperimentResult:
    """`models` restricts the candidates (default: all of `candidate_models`)."""
    if models is not None and (unknown := set(models) - set(candidate_models(random_state))):
        raise ValueError(f"no candidate model named {sorted(unknown)}")
    started_at = datetime.now(UTC)
    run_id = uuid.uuid4().hex
    commit = git_commit()
    dataset = experiment.load_dataset(dataset_path)
    builder = FEATURE_BUILDERS[experiment.feature_set]
    data = prepare_data(dataset, builder, random_state)
    if experiment.name != "synthetic_baseline":
        validate_external_evaluation_split(data, random_state)
    logger.info(
        "experiment=%s rows=%d patients=%s train=%d test=%d split=%s",
        experiment.name,
        len(data.labels),
        dataset.number_of_patients,
        len(data.split.train_index),
        len(data.split.test_index),
        data.split.strategy.value,
    )

    candidates = {
        name: _evaluate_candidate(name, model, data, random_state)
        for name, model in candidate_models(random_state).items()
        if models is None or name in models
    }
    selected = max(candidates, key=lambda name: _selection_key(candidates[name].cv_metrics))
    best = candidates[selected]

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME
    joblib.dump(best.model, model_path)
    metadata = ModelMetadata(
        experiment=experiment.name,
        model_type=selected,
        model_version=MODEL_VERSION,
        dataset_name=dataset.info.name,
        dataset_version=dataset.info.version,
        data_disclaimer=dataset.info.disclaimer,
        label_definition=dataset.info.label_definition,
        feature_set=builder.feature_set,
        features=list(builder.feature_names),
        classes=[str(c) for c in best.model.classes_],
        split_strategy=data.split.strategy.value,
        number_of_rows=len(data.labels),
        number_of_patients=dataset.number_of_patients,
        hyperparameters=hyperparameters(best.model),
        training_diagnostics=best.diagnostics,
        metrics=TrainingMetrics(
            held_out_test=best.test_metrics,
            cross_validation={name: c.cv_metrics for name, c in candidates.items()},
        ),
        training_date=started_at.isoformat(),
        git_commit=commit,
        artifact_sha256=file_sha256(model_path),
        random_state=random_state,
    )
    write_metadata(model_dir, metadata)

    files = [
        write_record(
            BenchmarkRecord(
                run_id=run_id,
                timestamp=started_at.isoformat(),
                git_commit=commit,
                git_dirty=commit.endswith("-dirty"),
                experiment=experiment.name,
                dataset_name=dataset.info.name,
                dataset_version=dataset.info.version,
                data_disclaimer=dataset.info.disclaimer,
                label_definition=dataset.info.label_definition,
                number_of_rows=len(data.labels),
                number_of_patients=dataset.number_of_patients,
                train_rows=len(data.split.train_index),
                test_rows=len(data.split.test_index),
                feature_set=builder.feature_set,
                features=list(builder.feature_names),
                split_strategy=data.split.strategy.value,
                random_state=random_state,
                model_name=name,
                hyperparameters=hyperparameters(result.model),
                selection_criterion=SELECTION_CRITERION,
                selected_for_deployment=name == selected,
                cross_validation_metrics=result.cv_metrics,
                test_metrics=result.test_metrics,
                training_time_seconds=result.training_time_seconds,
                inference_time_seconds=result.inference_time_seconds,
                inference_time_ms_per_row=result.inference_time_ms_per_row,
                training_diagnostics=result.diagnostics,
            ),
            benchmark_dir,
            started_at,
        )
        for name, result in candidates.items()
    ]
    return ExperimentResult(experiment.name, selected, candidates, metadata, files)


def log_result(result: ExperimentResult) -> None:
    for name, candidate in result.candidates.items():
        cv = candidate.cv_metrics
        logger.info(
            "cv %-20s macro_f1=%.4f emergency_recall=%.4f under_triage=%.4f",
            name,
            cv.macro_f1,
            cv.emergency_recall,
            cv.under_triage_rate,
        )
    test = result.candidates[result.selected_model].test_metrics
    logger.info(
        "selected=%s (by CV) held-out accuracy=%.4f macro_f1=%.4f emergency_recall=%.4f "
        "under_triage=%.4f over_triage=%.4f critical_under_triage=%.4f",
        result.selected_model,
        test.accuracy,
        test.macro_f1,
        test.emergency_recall,
        test.under_triage_rate,
        test.over_triage_rate,
        test.critical_under_triage_rate,
    )
    logger.info("benchmark records: %s", ", ".join(str(p) for p in result.benchmark_files))
    logger.info(result.metadata.data_disclaimer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train routing classifiers for an experiment")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="synthetic_baseline")
    parser.add_argument("--dataset", type=Path, help="defaults to the experiment's dataset")
    parser.add_argument("--model-dir", type=Path, help="defaults to the experiment's model dir")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    experiment = EXPERIMENTS[args.experiment]
    result = run_experiment(
        experiment,
        args.dataset or experiment.default_dataset,
        args.model_dir or experiment.default_model_dir,
        args.benchmark_dir,
    )
    log_result(result)


if __name__ == "__main__":
    main()
