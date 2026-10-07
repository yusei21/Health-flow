import numpy as np
import pytest

from app.ml.metrics import compute_metrics, triage_rates

P, U, E = "PRIMARY_CARE", "URGENT_CARE", "EMERGENCY"


def test_under_and_over_triage_follow_severity_order() -> None:
    y_true = np.array([P, U, E, E, P, U])
    y_pred = np.array([P, P, U, E, E, E])  # under: U→P, E→U ; over: P→E, U→E
    rates = triage_rates(y_true, y_pred)
    assert rates["under_triage_rate"] == pytest.approx(2 / 6, abs=1e-4)
    assert rates["over_triage_rate"] == pytest.approx(2 / 6, abs=1e-4)


def test_critical_under_triage_counts_only_true_emergencies() -> None:
    y_true = np.array([E, E, E, E, P, U])
    y_pred = np.array([E, U, P, E, P, P])
    assert triage_rates(y_true, y_pred)["critical_under_triage_rate"] == 0.5


def test_perfect_predictions_have_zero_triage_errors() -> None:
    y = np.array([P, U, E])
    assert triage_rates(y, y) == {
        "under_triage_rate": 0.0,
        "over_triage_rate": 0.0,
        "critical_under_triage_rate": 0.0,
    }


def test_no_emergency_in_labels_gives_zero_critical_rate() -> None:
    assert triage_rates(np.array([P, U]), np.array([P, P]))["critical_under_triage_rate"] == 0.0


def test_compute_metrics_is_consistent_with_emergency_recall() -> None:
    y_true = np.array([E, E, E, E, P, U, U, P])
    y_pred = np.array([E, U, P, E, P, U, E, U])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.critical_under_triage_rate == pytest.approx(1 - metrics.emergency_recall)
    assert metrics.confusion_matrix == [[1, 1, 0], [0, 1, 1], [1, 1, 2]]
