"""Registered ML experiments: which dataset, which feature set, where artifacts go.

Experiment C (MIMIC chief complaint → local LLM → SymptomExtraction → Health-flow
features) is planned and intentionally not registered yet. The same future path applies
to the Triagegeist chief complaint; only its structured baseline is registered here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.ml.data.base import read_canonical_csv
from app.ml.data.schemas import TrainingDataset
from app.ml.data.synthetic import DEFAULT_DATASET_PATH, load_synthetic_dataset
from app.ml.feature_builders import (
    HealthFlowSymptomFeatureBuilder,
    MimicStructuredFeatureBuilder,
    TriagegeistStructuredFeatureBuilder,
)

MIMIC_PROCESSED_PATH = Path("data/processed/routing_mimic_v1.csv")
TRIAGEGEIST_PROCESSED_PATH = Path("data/processed/routing_triagegeist_v1.csv")


@dataclass(frozen=True)
class Experiment:
    name: str
    feature_set: str
    default_dataset: Path
    default_model_dir: Path
    load_dataset: Callable[[Path], TrainingDataset]


EXPERIMENTS: dict[str, Experiment] = {
    experiment.name: experiment
    for experiment in (
        Experiment(
            name="synthetic_baseline",
            feature_set=HealthFlowSymptomFeatureBuilder.feature_set,
            default_dataset=DEFAULT_DATASET_PATH,
            default_model_dir=Path("models/synthetic-v1"),
            load_dataset=load_synthetic_dataset,
        ),
        Experiment(
            name="structured_mimic_baseline",
            feature_set=MimicStructuredFeatureBuilder.feature_set,
            default_dataset=MIMIC_PROCESSED_PATH,
            default_model_dir=Path("models/mimic-structured-v1"),
            load_dataset=read_canonical_csv,
        ),
        Experiment(
            name="structured_triagegeist_baseline",
            feature_set=TriagegeistStructuredFeatureBuilder.feature_set,
            default_dataset=TRIAGEGEIST_PROCESSED_PATH,
            default_model_dir=Path("models/triagegeist-structured-v1"),
            load_dataset=read_canonical_csv,
        ),
    )
}
