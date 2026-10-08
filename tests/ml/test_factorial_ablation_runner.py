"""Regression checks for offline ablation wiring, not clinical evaluation."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from benchmarks.scripts.run_factorial_ablation import load_cases, run_case


def sample() -> dict:
    return {
        "case_id": "synthetic-case-001",
        "report": "estou com tosse leve",
        "extraction": {"symptoms": ["cough"], "severity": "mild"},
        "reference_level": "PRIMARY_CARE",
        "_hash": "synthetic-fixture",
        "patient_record": {
            "patient_id": str(uuid4()),
            "user_id": str(uuid4()),
            "display_name": "Pessoa fictícia",
            "age": 67,
            "conditions": [{"name": "diabetes", "risk_factor": "diabetes"}],
        },
    }


def test_rule_ablation_same_decision_with_and_without_harness() -> None:
    case = sample()
    results = [
        asyncio.run(run_case(case, harness, False, context, None))
        for harness in (False, True)
        for context in (False, True)
    ]
    assert len(results) == 4
    assert {row["predicted_final"] for row in results} == {"PRIMARY_CARE"}
    assert {row["status"] for row in results} == {"ok"}
    assert {row["patient_context_enabled"] for row in results} == {False, True}


def test_precheck_emergency_bypasses_frozen_extraction_and_classifier() -> None:
    case = sample()
    case["report"] = "ele teve uma convulsão agora"
    for harness in (False, True):
        result = asyncio.run(run_case(case, harness, False, True, None))
        assert result["predicted_final"] == "EMERGENCY"
        assert result["llm_calls"] == 0


def test_corpus_rejects_duplicate_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case = sample()
    case.pop("_hash")
    path.write_text(
        json.dumps(case) + "\n" + json.dumps(case) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_cases(path)
