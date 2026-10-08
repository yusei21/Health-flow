"""Validate the illustrative, fictional 60-case factorial smoke pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from benchmarks.scripts.run_factorial_ablation import load_cases

EXPECTED = {"PRIMARY_CARE": 20, "URGENT_CARE": 20, "EMERGENCY": 20}
PROVENANCE = "illustrative_template_not_independent_clinical_label"


def validate_pilot(path: Path) -> dict[str, object]:
    cases = load_cases(path)
    counts = Counter(str(case["reference_level"]) for case in cases)
    if len(cases) != 60 or dict(counts) != EXPECTED:
        raise ValueError(\n            f"Expected 60 cases and 20 per class; found {len(cases)}: {dict(counts)}"\n        )
    if any(case.get("reference_provenance") != PROVENANCE for case in cases):
        raise ValueError("Every reference label must declare illustrative provenance")
    if any("patient_record" not in case for case in cases):
        raise ValueError("Every case must include a fictional patient record")
    profiles: Counter[str] = Counter()
    for case in cases:
        record = case["patient_record"]
        if not isinstance(record, dict):
            raise ValueError("patient_record must be a JSON object")
        if record.get("age") == 31:
            profiles["adult"] += 1
        elif record.get("age") == 76:
            profiles["elderly"] += 1
        else:
            raise ValueError("Unknown fixture age; use only fictional fixed profiles")
    if profiles != {"adult": 30, "elderly": 30}:
        raise ValueError(f"Unexpected profile distribution: {dict(profiles)}")
    return {
        "status": "synthetic_pilot_fixture_valid",
        "n_cases": len(cases),
        "reference_support": dict(counts),
        "profiles": dict(profiles),
        "limitations": [
            "Labels are illustrative templates, not independently adjudicated",
            "Cases may resemble synthetic training patterns and are not held-out clinical evidence",
            "No real EHR or production LLM is used",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(validate_pilot(args.jsonl), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
