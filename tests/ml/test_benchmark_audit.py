import numpy as np
import pytest

from app.ml.splits import split_train_test
from app.ml.training.train import PreparedData, validate_external_evaluation_split
from benchmarks.scripts.audit_results import audit_record, wilson_interval


def test_wilson_interval_contains_observed_rate() -> None:
    interval = wilson_interval(13, 23)
    assert interval is not None
    assert interval[0] < 13 / 23 < interval[1]


def test_audit_flags_absent_primary_class() -> None:
    matrix = [[0, 0, 0], [3, 7, 8], [1, 9, 13]]
    record = {
        "experiment": "structured_mimic_baseline",
        "dataset_version": "demo",
        "run_id": "run",
        "model_name": "neural_network_mlp",
        "selected_for_deployment": True,
        "number_of_patients": 56,
        "test_rows": 41,
        "git_dirty": False,
        "test_metrics": {
            "accuracy": 0.4878,
            "macro_f1": 0.3342,
            "critical_under_triage_rate": 0.4348,
            "confusion_matrix_labels": ["PRIMARY_CARE", "URGENT_CARE", "EMERGENCY"],
            "confusion_matrix": matrix,
            "per_class": {
                "PRIMARY_CARE": {"support": 0},
                "URGENT_CARE": {"support": 18},
                "EMERGENCY": {"support": 23},
            },
        },
    }
    audit = audit_record(record)
    assert audit.emergency_support == 23
    assert audit.emergency_missed == 10
    assert "TEST_CLASS_ABSENT:PRIMARY_CARE" in audit.flags


def test_external_split_rejects_rare_class() -> None:
    labels = np.array(["PRIMARY_CARE"] * 2 + ["URGENT_CARE"] * 40 + ["EMERGENCY"] * 40)
    split = split_train_test(labels, None, random_state=42)
    data = PreparedData(
        features=np.zeros((len(labels), 1)),
        labels=labels,
        groups=None,
        split=split,
    )
    with pytest.raises(ValueError, match="Insufficient support"):
        validate_external_evaluation_split(data, random_state=42)
