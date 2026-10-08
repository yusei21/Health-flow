"""Check fixture invariants without interpreting synthetic labels clinically."""

from pathlib import Path

import pytest

from benchmarks.scripts.validate_factorial_pilot import validate_pilot

FIXTURE = Path("benchmarks/fixtures/ablacao_piloto.jsonl")


def test_pilot_has_balanced_fictional_cases() -> None:
    result = validate_pilot(FIXTURE)
    assert result["n_cases"] == 60
    assert result["reference_support"] == {
        "PRIMARY_CARE": 20,
        "URGENT_CARE": 20,
        "EMERGENCY": 20,
    }
    assert result["profiles"] == {"adult": 30, "elderly": 30}


def test_pilot_rejects_missing_data(tmp_path: Path) -> None:
    missing = tmp_path / "not-found.jsonl"
    with pytest.raises(FileNotFoundError):
        validate_pilot(missing)
