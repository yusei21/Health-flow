"""Classification metrics for the routing model.

Accuracy alone is misleading with imbalanced classes, so per-class precision/recall/F1,
macro averages, the confusion matrix and EMERGENCY recall are always reported.

Triage-direction metrics use the severity order PRIMARY_CARE(0) < URGENT_CARE(1) <
EMERGENCY(2):
- under_triage_rate: share of ALL cases predicted less severe than the label;
- over_triage_rate: share of ALL cases predicted more severe than the label;
- critical_under_triage_rate: share of true EMERGENCY cases predicted as anything else
  (= 1 - EMERGENCY recall), the error the system can least afford.
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.schemas.care import CareLevel

CLASS_ORDER = [level.value for level in CareLevel]


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class ClassificationMetrics(BaseModel):
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    emergency_recall: float
    under_triage_rate: float
    over_triage_rate: float
    critical_under_triage_rate: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix_labels: list[str]
    confusion_matrix: list[list[int]]  # rows = true label, columns = predicted label


def compute_metrics(y_true: NDArray[np.str_], y_pred: NDArray[np.str_]) -> ClassificationMetrics:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    per_class = {
        label: ClassMetrics(
            precision=round(float(precision[i]), 4),
            recall=round(float(recall[i]), 4),
            f1=round(float(f1[i]), 4),
            support=int(support[i]),
        )
        for i, label in enumerate(CLASS_ORDER)
    }
    return ClassificationMetrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        macro_precision=round(float(np.mean(precision)), 4),
        macro_recall=round(float(np.mean(recall)), 4),
        macro_f1=round(float(np.mean(f1)), 4),
        emergency_recall=per_class[CareLevel.EMERGENCY.value].recall,
        **triage_rates(y_true, y_pred),
        per_class=per_class,
        confusion_matrix_labels=CLASS_ORDER,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    )


def triage_rates(y_true: NDArray[np.str_], y_pred: NDArray[np.str_]) -> dict[str, float]:
    true_rank = np.array([CareLevel(label).rank for label in y_true])
    pred_rank = np.array([CareLevel(label).rank for label in y_pred])
    emergency = true_rank == CareLevel.EMERGENCY.rank
    total = max(len(true_rank), 1)
    return {
        "under_triage_rate": round(float(np.sum(pred_rank < true_rank)) / total, 4),
        "over_triage_rate": round(float(np.sum(pred_rank > true_rank)) / total, 4),
        "critical_under_triage_rate": (
            round(float(np.mean(pred_rank[emergency] != true_rank[emergency])), 4)
            if emergency.any()
            else 0.0
        ),
    }


class TrainingMetrics(BaseModel):
    held_out_test: ClassificationMetrics
    cross_validation: dict[str, ClassificationMetrics]
