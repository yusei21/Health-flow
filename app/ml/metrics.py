"""Classification metrics for the routing model.

Accuracy alone is misleading with imbalanced classes, so per-class precision/recall/F1,
macro averages, the confusion matrix and EMERGENCY recall are always reported.
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
        per_class=per_class,
        confusion_matrix_labels=CLASS_ORDER,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    )


class TrainingMetrics(BaseModel):
    held_out_test: ClassificationMetrics
    cross_validation: dict[str, ClassificationMetrics]
