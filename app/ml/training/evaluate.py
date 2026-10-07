"""Evaluate a saved model on its held-out test split.

The split is recomputed deterministically from the dataset and the model's
`random_state`, so only rows the model never trained on are scored. If the dataset
changed since training, evaluation refuses to run instead of silently mixing splits.
"""

import argparse
import logging
from pathlib import Path

from app.ml.classifier import RoutingClassifier
from app.ml.experiments import EXPERIMENTS, Experiment
from app.ml.feature_builders import FEATURE_BUILDERS
from app.ml.metrics import ClassificationMetrics, compute_metrics
from app.ml.training.train import prepare_data

logger = logging.getLogger(__name__)


def evaluate_experiment(
    experiment: Experiment, dataset_path: Path, model_dir: Path
) -> ClassificationMetrics:
    classifier = RoutingClassifier.load(model_dir, expected_feature_set=experiment.feature_set)
    meta = classifier.metadata
    dataset = experiment.load_dataset(dataset_path)
    if (dataset.info.version, len(dataset.examples)) != (meta.dataset_version, meta.number_of_rows):
        raise ValueError("dataset differs from the one used in training; retrain first")
    data = prepare_data(dataset, FEATURE_BUILDERS[meta.feature_set], meta.random_state)
    if data.split.strategy.value != meta.split_strategy:
        raise ValueError("split strategy differs from training")
    test = data.split.test_index
    return compute_metrics(data.labels[test], classifier.predict_labels(data.features[test]))


def report(metrics: ClassificationMetrics) -> None:
    logger.info("accuracy                    %.4f", metrics.accuracy)
    logger.info(
        "macro precision / recall    %.4f / %.4f", metrics.macro_precision, metrics.macro_recall
    )
    logger.info("macro F1                    %.4f", metrics.macro_f1)
    for label, cls in metrics.per_class.items():
        logger.info(
            "  %-13s precision=%.4f recall=%.4f f1=%.4f support=%d",
            label,
            cls.precision,
            cls.recall,
            cls.f1,
            cls.support,
        )
    logger.info("emergency recall            %.4f", metrics.emergency_recall)
    logger.info("under-triage rate           %.4f", metrics.under_triage_rate)
    logger.info("over-triage rate            %.4f", metrics.over_triage_rate)
    logger.info("critical under-triage rate  %.4f", metrics.critical_under_triage_rate)
    logger.info("confusion matrix (rows=true, cols=pred) %s", metrics.confusion_matrix_labels)
    for label, row in zip(metrics.confusion_matrix_labels, metrics.confusion_matrix, strict=True):
        logger.info("  %-13s %s", label, row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on its held-out test")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="synthetic_baseline")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    experiment = EXPERIMENTS[args.experiment]
    metrics = evaluate_experiment(
        experiment,
        args.dataset or experiment.default_dataset,
        args.model_dir or experiment.default_model_dir,
    )
    report(metrics)


if __name__ == "__main__":
    main()
