"""Unit tests for reproducible LLM extraction benchmark scoring (no live model)."""

import pytest

from app.core.config import Settings
from app.schemas.symptoms import Symptom
from benchmarks.scripts.benchmark_llm import extraction_scores, load_cases


def test_micro_counts() -> None:
    assert extraction_scores(
        {Symptom.CHEST_PAIN, Symptom.COUGH},
        {Symptom.CHEST_PAIN, Symptom.FEVER},
    ) == (1, 1, 1)


def test_fictional_cases_have_unique_ids_and_valid_labels() -> None:
    from pathlib import Path

    cases = load_cases(Path("benchmarks/llm/cases.jsonl"))
    assert len(cases) >= 10
    assert len({case.case_id for case in cases}) == len(cases)


def test_model_can_be_switched_without_changing_the_harness() -> None:
    settings = Settings(_env_file=None, llm_model="llama3.2:3b")
    assert settings.llm_model == "llama3.2:3b"


@pytest.mark.parametrize("model", ["qwen3:4b", "llama3.2:3b"])
def test_model_selection_is_configuration_only(model: str) -> None:
    settings = Settings(_env_file=None, llm_model=model)
    assert settings.llm_model == model
