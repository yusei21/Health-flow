import logging

import numpy as np

from app.core.exceptions import MLInferenceError
from app.ml.classifier import RoutingClassifier
from app.ml.features import build_features
from app.schemas.care import CareLevel
from app.schemas.patient import PatientContext
from app.schemas.routing import MLPrediction
from app.schemas.symptoms import SymptomExtraction

logger = logging.getLogger(__name__)


class RoutingInferenceService:
    """Auxiliary probabilistic classification. Never the sole authority."""

    def __init__(self, classifier: RoutingClassifier | None) -> None:
        self._classifier = classifier

    @property
    def available(self) -> bool:
        return self._classifier is not None

    def predict(self, extraction: SymptomExtraction, context: PatientContext) -> MLPrediction:
        if self._classifier is None:
            raise MLInferenceError("no model loaded")
        features = np.array([build_features(extraction, context)], dtype=np.float64)
        try:
            probabilities = self._classifier.predict_proba(features)[0]
            by_level = {
                CareLevel(label): round(float(p), 4)
                for label, p in zip(self._classifier.classes, probabilities, strict=True)
            }
        except (ValueError, TypeError) as exc:
            raise MLInferenceError("prediction failed") from exc
        predicted = max(by_level, key=lambda level: by_level[level])
        return MLPrediction(
            predicted_class=predicted,
            confidence=by_level[predicted],
            probabilities=by_level,
            model_version=self._classifier.metadata.version,
        )
