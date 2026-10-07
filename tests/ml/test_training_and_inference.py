import json
import shutil
from pathlib import Path

import pytest

from app.core.exceptions import MLInferenceError, MLModelUnavailableError
from app.ml.classifier import METADATA_FILENAME, MODEL_FILENAME, RoutingClassifier
from app.ml.inference import RoutingInferenceService
from app.ml.training.dataset import (
    DATA_DISCLAIMER,
    generate_examples,
    load_dataset,
    save_dataset,
)
from app.schemas.care import CareLevel
from app.schemas.patient import AgeRange, PatientContext
from app.schemas.symptoms import Severity, Symptom, SymptomExtraction


def test_dataset_generation_is_reproducible_and_round_trips(tmp_path: Path) -> None:
    assert generate_examples(50, seed=1) == generate_examples(50, seed=1)
    path = tmp_path / "d.csv"
    examples = generate_examples(50, seed=1)
    save_dataset(examples, path)
    assert load_dataset(path) == examples
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["disclaimer"] == DATA_DISCLAIMER


def test_dataset_contains_all_classes() -> None:
    labels = {ex.label for ex in generate_examples(500, seed=3)}
    assert labels == set(CareLevel)


def test_load_dataset_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="columns"):
        load_dataset(path)


def test_metadata_contains_required_fields(trained_model_dir: Path) -> None:
    meta = json.loads((trained_model_dir / METADATA_FILENAME).read_text())
    for key in ("model_type", "features", "training_date", "metrics", "version", "dataset_version"):
        assert key in meta
    assert meta["data_disclaimer"] == DATA_DISCLAIMER
    held_out = meta["metrics"]["held_out_test"]
    assert {"accuracy", "macro_f1", "emergency_recall", "confusion_matrix"} <= held_out.keys()
    assert set(meta["metrics"]["cross_validation"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_inference_returns_normalized_probabilities(classifier: RoutingClassifier) -> None:
    prediction = RoutingInferenceService(classifier).predict(
        SymptomExtraction(symptoms=[Symptom.COUGH], severity=Severity.MILD, duration_minutes=4320),
        PatientContext(age_range=AgeRange.ADULT),
    )
    assert set(prediction.probabilities) == set(CareLevel)
    assert sum(prediction.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert prediction.confidence == max(prediction.probabilities.values())


def test_model_learned_synthetic_pattern(classifier: RoutingClassifier) -> None:
    # Sanity check against the synthetic generator only; not clinical evidence.
    service = RoutingInferenceService(classifier)
    mild = service.predict(
        SymptomExtraction(
            symptoms=[Symptom.RUNNY_NOSE, Symptom.SORE_THROAT],
            severity=Severity.MILD,
            duration_minutes=5 * 24 * 60,
        ),
        PatientContext(age_range=AgeRange.ADULT),
    )
    severe = service.predict(
        SymptomExtraction(
            symptoms=[Symptom.SEIZURE, Symptom.LOSS_OF_CONSCIOUSNESS],
            severity=Severity.SEVERE,
            duration_minutes=10,
        ),
        PatientContext(age_range=AgeRange.ADULT),
    )
    assert mild.predicted_class is CareLevel.PRIMARY_CARE
    assert severe.predicted_class is CareLevel.EMERGENCY


def test_missing_artifact_raises_unavailable(tmp_path: Path) -> None:
    with pytest.raises(MLModelUnavailableError):
        RoutingClassifier.load(tmp_path)


def test_tampered_artifact_is_refused(trained_model_dir: Path, tmp_path: Path) -> None:
    shutil.copytree(trained_model_dir, tmp_path / "m")
    with (tmp_path / "m" / MODEL_FILENAME).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(MLModelUnavailableError, match="checksum"):
        RoutingClassifier.load(tmp_path / "m")


def test_feature_mismatch_is_refused(trained_model_dir: Path, tmp_path: Path) -> None:
    shutil.copytree(trained_model_dir, tmp_path / "m")
    meta_path = tmp_path / "m" / METADATA_FILENAME
    meta = json.loads(meta_path.read_text())
    meta["features"] = meta["features"][:-1]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(MLModelUnavailableError, match="features"):
        RoutingClassifier.load(tmp_path / "m")


def test_service_without_model_raises_inference_error() -> None:
    with pytest.raises(MLInferenceError):
        RoutingInferenceService(None).predict(SymptomExtraction(), PatientContext())
