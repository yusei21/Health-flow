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
from app.ml.features import FEATURE_NAMES
from app.ml.metrics import TrainingMetrics

MODEL_FILENAME = "routing_model.joblib"
METADATA_FILENAME = "metadata.json"


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_type: str
    version: str
    features: list[str]
    classes: list[str]
    training_date: str
    dataset_version: str
    data_disclaimer: str
    artifact_sha256: str
    metrics: TrainingMetrics
    random_state: int


class _ProbabilisticEstimator(Protocol):
    classes_: NDArray[np.str_]

    def predict_proba(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...


class RoutingClassifier:
    def __init__(self, estimator: _ProbabilisticEstimator, metadata: ModelMetadata) -> None:
        self._estimator = estimator
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: Path) -> Self:
        model_path = model_dir / MODEL_FILENAME
        metadata_path = model_dir / METADATA_FILENAME
        if not model_path.is_file() or not metadata_path.is_file():
            raise MLModelUnavailableError(f"model artifact not found in {model_dir}")
        try:
            metadata = ModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise MLModelUnavailableError("invalid model metadata") from exc
        if tuple(metadata.features) != FEATURE_NAMES:
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
