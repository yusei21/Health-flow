"""Development-only paired context and Harness failure diagnostics."""

import asyncio
from pathlib import Path

from app.ml.classifier import RoutingClassifier
from benchmarks.scripts.analyze_context_sensitivity import analyze
from benchmarks.scripts.evaluate_harness_resilience import evaluate
from benchmarks.scripts.run_factorial_ablation import load_cases


def test_context_sensitivity_covers_all_fictional_cases(trained_model_dir: Path) -> None:
    model = RoutingClassifier.load(trained_model_dir)
    report = analyze(load_cases(Path("benchmarks/fixtures/ablacao_piloto.jsonl")), model)
    assert report["n_cases"] == 60
    assert len(report["per_case"]) == 60
    assert 0 <= report["predicted_classes_changed"] <= 60
    assert 0 <= report["probability_vectors_changed"] <= 60


def test_harness_failure_injection_uses_real_components() -> None:
    results = asyncio.run(evaluate())
    by_failure = {row["injected_failure"]: row for row in results["trials"]}
    assert by_failure["none"]["harness"]["status"] == "completed"
    assert by_failure["ml"]["harness"]["status"] == "completed"
    assert "MLInferenceError" in by_failure["ml"]["harness"]["handled_errors"]
    assert by_failure["ml"]["direct"]["error_type"] == "MLInferenceError"
    assert by_failure["facilities"]["harness"]["status"] == "completed"
    assert by_failure["facilities"]["direct"]["error_type"] == "FacilityProviderError"
    assert by_failure["llm"]["harness"]["error_type"] == "LLMUnavailableError"
    assert by_failure["policy"]["harness"]["error_type"] == "HarnessPolicyError"
