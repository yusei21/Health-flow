"""Train, compare and persist the routing classifier.

Never runs inside the API. Usage: `python -m app.ml.training.train` (or `make train`).
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.ml.classifier import MODEL_FILENAME, ModelMetadata, file_sha256, write_metadata
from app.ml.features import FEATURE_NAMES, build_features
from app.ml.metrics import ClassificationMetrics, TrainingMetrics, compute_metrics
from app.ml.training.dataset import (
    DATA_DISCLAIMER,
    DATASET_VERSION,
    DEFAULT_DATASET_PATH,
    RoutingExample,
    load_dataset,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
MODEL_VERSION = "0.1.0"


def candidate_models() -> dict[str, ClassifierMixin | Pipeline]:
    # class_weight="balanced" because EMERGENCY is the minority class and the one
    # whose recall matters most.
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


@dataclass(frozen=True)
class TrainingResult:
    selected_model: str
    cv_metrics: dict[str, ClassificationMetrics]
    test_metrics: ClassificationMetrics
    metadata: ModelMetadata


def to_arrays(examples: list[RoutingExample]) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    features = np.array([build_features(ex.extraction, ex.context) for ex in examples])
    labels = np.array([ex.label.value for ex in examples])
    return features, labels


def _selection_key(metrics: ClassificationMetrics) -> tuple[float, float]:
    # Prefer EMERGENCY recall (rounded so noise does not dominate), then macro F1.
    return (round(metrics.emergency_recall, 2), metrics.macro_f1)


def train(
    dataset_path: Path, output_dir: Path, dataset_version: str = DATASET_VERSION
) -> TrainingResult:
    features, labels = to_arrays(load_dataset(dataset_path))
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_metrics: dict[str, ClassificationMetrics] = {}
    for name, model in candidate_models().items():
        predictions = cross_val_predict(model, x_train, y_train, cv=folds)
        cv_metrics[name] = compute_metrics(y_train, predictions)

    selected = max(cv_metrics, key=lambda name: _selection_key(cv_metrics[name]))
    model = candidate_models()[selected].fit(x_train, y_train)
    test_metrics = compute_metrics(y_test, model.predict(x_test))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    joblib.dump(model, model_path)
    metadata = ModelMetadata(
        model_type=selected,
        version=MODEL_VERSION,
        features=list(FEATURE_NAMES),
        classes=[str(c) for c in model.classes_],
        training_date=datetime.now(UTC).isoformat(),
        dataset_version=dataset_version,
        data_disclaimer=DATA_DISCLAIMER,
        artifact_sha256=file_sha256(model_path),
        metrics=TrainingMetrics(held_out_test=test_metrics, cross_validation=cv_metrics),
        random_state=RANDOM_STATE,
    )
    write_metadata(output_dir, metadata)
    return TrainingResult(selected, cv_metrics, test_metrics, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the routing classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = train(args.dataset, args.output_dir, args.dataset_version)
    for name, cv in result.cv_metrics.items():
        logger.info(
            "cv %-20s macro_f1=%.4f emergency_recall=%.4f", name, cv.macro_f1, cv.emergency_recall
        )
    test = result.test_metrics
    logger.info(
        "selected=%s held-out accuracy=%.4f macro_f1=%.4f emergency_recall=%.4f",
        result.selected_model,
        test.accuracy,
        test.macro_f1,
        test.emergency_recall,
    )
    logger.info("confusion matrix %s (rows=true, cols=pred):", test.confusion_matrix_labels)
    for label, row in zip(test.confusion_matrix_labels, test.confusion_matrix, strict=True):
        logger.info("  %-13s %s", label, row)
    logger.info(DATA_DISCLAIMER)


if __name__ == "__main__":
    main()
