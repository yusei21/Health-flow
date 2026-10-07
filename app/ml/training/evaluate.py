"""Re-evaluate a saved routing model on any dataset following the CSV schema."""

import argparse
from pathlib import Path

import numpy as np

from app.ml.classifier import RoutingClassifier
from app.ml.features import build_features
from app.ml.metrics import ClassificationMetrics, compute_metrics
from app.ml.training.dataset import load_dataset


def evaluate_saved_model(model_dir: Path, dataset_path: Path) -> ClassificationMetrics:
    classifier = RoutingClassifier.load(model_dir)
    examples = load_dataset(dataset_path)
    features = np.array([build_features(ex.extraction, ex.context) for ex in examples])
    y_true = np.array([ex.label.value for ex in examples])
    return compute_metrics(y_true, classifier.predict_labels(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved routing model on a dataset")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_saved_model(args.model_dir, args.dataset).model_dump_json(indent=2))  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
