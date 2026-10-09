"""Paired ML-context sensitivity on frozen fictional patient cases.

Outputs engineering diagnostics only; not clinical validation or real EHR evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any
from uuid import uuid5

from app.ml.classifier import RoutingClassifier
from app.ml.features import build_features
from app.ml.inference import RoutingInferenceService
from app.schemas.care import CareLevel
from app.schemas.patient import PatientRecord
from app.schemas.symptoms import SymptomExtraction
from benchmarks.scripts.run_factorial_ablation import (
    NAMESPACE,
    FrozenContext,
    load_cases,
)

CLASSES = tuple(CareLevel)


def analyze(cases: list[dict[str, Any]], classifier: RoutingClassifier) -> dict[str, Any]:
    inference = RoutingInferenceService(classifier)
    changed = 0
    nonzero_features = 0
    nonzero_probabilities = 0
    per_case: list[dict[str, Any]] = []
    deltas: list[float] = []
    for case in cases:
        extraction = SymptomExtraction.model_validate(case["extraction"])
        user_id = uuid5(NAMESPACE, str(case["case_id"]))
        raw_record = case.get("patient_record")
        record = (
            PatientRecord.model_validate({**raw_record, "user_id": user_id})
            if raw_record is not None
            else None
        )
        # FrozenContext does not perform I/O: use exactly the same builder as ablation.
        empty = FrozenContext(record, False).builder.build(None, extraction)
        full = FrozenContext(record, True).builder.build(record, extraction)
        features_changed = build_features(extraction, empty) != build_features(extraction, full)
        before = inference.predict(extraction, empty)
        after = inference.predict(extraction, full)
        diff = max(
            abs(before.probabilities.get(level, 0.0) - after.probabilities.get(level, 0.0))
            for level in CLASSES
        )
        changed += before.predicted_class != after.predicted_class
        nonzero_features += features_changed
        nonzero_probabilities += diff > 0
        deltas.append(diff)
        per_case.append(
            {
                "case_id": str(case["case_id"]),
                "age_range_without": empty.age_range.value,
                "age_range_with": full.age_range.value,
                "risk_factors_with": sorted(item.value for item in full.risk_factors),
                "features_changed": features_changed,
                "predicted_without": before.predicted_class.value,
                "predicted_with": after.predicted_class.value,
                "probability_delta_max": round(diff, 4),
            }
        )
    return {
        "status": "offline_context_sensitivity_diagnostic",
        "n_cases": len(cases),
        "feature_vectors_changed": nonzero_features,
        "probability_vectors_changed": nonzero_probabilities,
        "predicted_classes_changed": changed,
        "max_probability_delta": max(deltas, default=0),
        "median_probability_delta": statistics.median(deltas) if deltas else None,
        "per_case": per_case,
        "limitations": [
            "Uses a synthetic model and fictional chart context",
            "Analyzes model predictions, not final SafetyEngine routing decisions",
            "No adjudicated clinical reference labels",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.cases.resolve():
        parser.error("Input and output must differ")
    report = analyze(load_cases(args.cases), RoutingClassifier.load(args.model_dir))
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sys.stdout.write(
        f"context features changed: {report['feature_vectors_changed']}; "
        f"probabilities changed: {report['probability_vectors_changed']}; "
        f"ML classes changed: {report['predicted_classes_changed']}\n"
    )


if __name__ == "__main__":
    main()
