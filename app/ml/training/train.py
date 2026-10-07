"""Train, compare and persist routing classifiers for a registered experiment.

Never runs inside the API. Usage:
    python -m app.ml.training.train --experiment synthetic_baseline
    python -m app.ml.training.train --experiment structured_mimic_baseline

Protocol: split train/test (grouped by patient when available) → cross-validate each
candidate on the training part only → select by CV → fit on the training part →
evaluate every candidate once on the held-out test for reporting. The test set never
influences selection.
"""

import argparse
import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.classifier import MODEL_FILENAME, ModelMetadata, file_sha256, write_metadata
from app.ml.data.schemas import TrainingDataset
from app.ml.experiments import EXPERIMENTS, Experiment
from app.ml.feature_builders import FEATURE_BUILDERS, FeatureBuilder
from app.ml.metrics import ClassificationMetrics, TrainingMetrics, compute_metrics
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
    return value if value is None or isinstance(value, bool | int | float | str) else repr(value)


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model: Pipeline
    cv_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    training_time_seconds: float
    inference_time_ms_per_row: float


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
    cv_pred = cross_val_predict(clone(model), x_train, y_train, cv=folds)

    started = time.perf_counter()
    fitted = clone(model).fit(x_train, y_train)
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    test_pred = fitted.predict(data.features[test_idx])
    inference_ms = (time.perf_counter() - started) * 1000 / max(len(test_idx), 1)

    return CandidateResult(
        name=name,
        model=fitted,
        cv_metrics=compute_metrics(y_train, cv_pred),
        test_metrics=compute_metrics(data.labels[test_idx], test_pred),
        training_time_seconds=round(training_time, 3),
        inference_time_ms_per_row=round(inference_ms, 5),
    )


def run_experiment(
    experiment: Experiment,
    dataset_path: Path,
    model_dir: Path,
    benchmark_dir: Path = DEFAULT_BENCHMARK_DIR,
    random_state: int = RANDOM_STATE,
) -> ExperimentResult:
    started_at = datetime.now(UTC)
    run_id = uuid.uuid4().hex
    commit = git_commit()
    dataset = experiment.load_dataset(dataset_path)
    builder = FEATURE_BUILDERS[experiment.feature_set]
    data = prepare_data(dataset, builder, random_state)
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
                inference_time_ms_per_row=result.inference_time_ms_per_row,
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
