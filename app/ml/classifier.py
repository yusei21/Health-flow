"""Loading and querying the persisted routing model."""

import hashlib
import json
from pathlib import Path
from typing import Protocol, Self

import joblib
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.exceptions import MLModelUnavailableError
from app.ml.feature_builders import FEATURE_BUILDERS
from app.ml.metrics import TrainingMetrics

MODEL_FILENAME = "routing_model.joblib"
METADATA_FILENAME = "metadata.json"


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    experiment: str
    model_type: str
    model_version: str
    dataset_name: str
    dataset_version: str
    data_disclaimer: str
    label_definition: str
    feature_set: str
    features: list[str]
    classes: list[str]
    split_strategy: str
    number_of_rows: int
    number_of_patients: int | None
    hyperparameters: dict[str, dict[str, object]]
    metrics: TrainingMetrics
    training_date: str
    git_commit: str
    artifact_sha256: str
    random_state: int


class _ProbabilisticEstimator(Protocol):
    classes_: NDArray[np.str_]

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


class RoutingClassifier:
    def __init__(self, estimator: _ProbabilisticEstimator, metadata: ModelMetadata) -> None:
        self._estimator = estimator
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: Path, expected_feature_set: str | None = None) -> Self:
        model_path = model_dir / MODEL_FILENAME
        metadata_path = model_dir / METADATA_FILENAME
        if not model_path.is_file() or not metadata_path.is_file():
            raise MLModelUnavailableError(f"model artifact not found in {model_dir}")
        try:
            metadata = ModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise MLModelUnavailableError("invalid model metadata") from exc
        if expected_feature_set is not None and metadata.feature_set != expected_feature_set:
            raise MLModelUnavailableError(
                f"model feature set {metadata.feature_set!r} != expected {expected_feature_set!r}"
            )
        builder = FEATURE_BUILDERS.get(metadata.feature_set)
        if builder is None or tuple(metadata.features) != builder.feature_names:
            raise MLModelUnavailableError("model features differ from current feature code")
        # joblib unpickles arbitrary code: only load the artifact we produced ourselves.
        if file_sha256(model_path) != metadata.artifact_sha256:
            raise MLModelUnavailableError("model artifact checksum mismatch")
        return cls(joblib.load(model_path), metadata)

    @property
    def classes(self) -> list[str]:
        return [str(label) for label in self._estimator.classes_]

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._estimator.predict_proba(features)

    def predict_labels(self, features: NDArray[np.float64]) -> NDArray[np.str_]:
        return np.asarray(self._estimator.classes_)[np.argmax(self.predict_proba(features), axis=1)]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metadata(model_dir: Path, metadata: ModelMetadata) -> None:
    (model_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
