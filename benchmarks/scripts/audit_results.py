"""Audit immutable benchmark JSONs without using test scores to choose a model.

Reports uncertainty for EMERGENCY recall using a Wilson binomial interval.
This is a descriptive interval conditional on the held-out sample, not a
patient-cluster-adjusted confidence interval or clinical validation.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.ml.metrics import CLASS_ORDER


class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str
    dataset_version: str
    run_id: str
    model: str
    selected_by_cv: bool
    test_rows: int
    patients: int | None
    accuracy: float
    macro_f1: float
    emergency_recall: float
    emergency_recall_wilson_95: tuple[float, float] | None
    emergency_support: int
    emergency_missed: int
    critical_under_triage: float
    flags: list[str]


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return (
        round(max(0.0, center - margin), 4),
        round(min(1.0, center + margin), 4),
    )


def audit_record(record: dict[str, object]) -> AuditResult:
    metrics = record["test_metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("test_metrics must be an object")
    labels = metrics["confusion_matrix_labels"]
    matrix = metrics["confusion_matrix"]
    if labels != CLASS_ORDER or not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError("Expected an ordered three-class confusion matrix")
    if any(not isinstance(row, list) or len(row) != 3 for row in matrix):
        raise ValueError("Invalid confusion matrix shape")
    if any(not isinstance(v, int) or v < 0 for row in matrix for v in row):
        raise ValueError("Confusion matrix must contain nonnegative integer counts")
    if sum(sum(row) for row in matrix) != record["test_rows"]:
        raise ValueError("Confusion matrix total disagrees with test_rows")
    emergency = matrix[2]
    support = sum(emergency)
    correct = emergency[2]
    per_class = metrics["per_class"]
    if not isinstance(per_class, dict):
        raise ValueError("per_class must be an object")
    flags: list[str] = []
    for idx, label in enumerate(CLASS_ORDER):
        observed = sum(matrix[idx])
        if observed == 0:
            flags.append(f"TEST_CLASS_ABSENT:{label}")
        elif observed < 10:
            flags.append(f"LOW_TEST_SUPPORT:{label}:{observed}")
        cls = per_class.get(label)
        if not isinstance(cls, dict) or cls.get("support") != observed:
            raise ValueError(f"Per-class support mismatch for {label}")
    if record.get("git_dirty") or str(record.get("git_commit", "")).endswith("-dirty"):
        flags.append("DIRTY_GIT_TREE")
    if record.get("dataset_name") == "synthetic":
        flags.append("SYNTHETIC_LABELS_NOT_CLINICAL_EVIDENCE")
    if support < 30:
        flags.append("SMALL_EMERGENCY_TEST_SUPPORT")
    if record.get("number_of_patients") is not None:
        flags.append("WILSON_INTERVAL_NOT_PATIENT_CLUSTER_ADJUSTED")
    return AuditResult(
        experiment=str(record["experiment"]),
        dataset_version=str(record["dataset_version"]),
        run_id=str(record["run_id"]),
        model=str(record["model_name"]),
        selected_by_cv=bool(record["selected_for_deployment"]),
        test_rows=int(record["test_rows"]),
        patients=(
            int(record["number_of_patients"])
            if record.get("number_of_patients") is not None
            else None
        ),
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        emergency_recall=round(correct / support, 4) if support else 0.0,
        emergency_recall_wilson_95=wilson_interval(correct, support),
        emergency_support=support,
        emergency_missed=support - correct,
        critical_under_triage=float(metrics["critical_under_triage_rate"]),
        flags=flags,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Immutable per-model benchmark JSONs"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/audit.json")
    )
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.paths]
    results = [audit_record(record) for record in records]
    grouping = Counter((result.experiment, result.run_id) for result in results)
    if len(grouping) != 1:
        parser.error("Supply only one experiment and one run_id per audit")
    if len({r.model for r in results}) != len(results):
        parser.error("Duplicate model records")
    if len({r.model for r in results}) != 4:
        parser.error("Expected all four candidate models in the same run")
    if sum(r.selected_by_cv for r in results) != 1:
        parser.error("Expected exactly one model selected by CV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "scope": "descriptive_model_only_not_full_harness",
                "interval_method": "Wilson 95%, row-level, not adjusted for patient clusters",
                "results": [r.model_dump(mode="json") for r in results],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    sys.stdout.write(f"Audit written: {args.output}\n")
    for result in results:
        sys.stdout.write(
            f"{result.model}: emergency_recall={result.emergency_recall:.4f} "
            f"n_emergency={result.emergency_support} missed={result.emergency_missed} "
            f"flags={','.join(result.flags) or 'none'}\n"
        )


if __name__ == "__main__":
    main()
