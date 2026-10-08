"""Reproducible evaluation of local LLM symptom extraction on labeled, fictional cases.

Does not measure routing quality or clinical safety. No user/patient records are read.
JSON output is immutable per run; individual reports and predictions are not persisted.
"""

import argparse
import asyncio
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.agents.intent_agent import IntentAgent
from app.core.config import Settings
from app.llm.ollama_provider import OllamaLLMProvider
from app.ml.training.benchmark import git_commit
from app.schemas.symptoms import Severity, Symptom


class LabeledCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    report: str = Field(min_length=1)
    symptoms: list[Symptom]
    severity: Severity = Severity.UNKNOWN
    duration_minutes: int | None = None
    age: int | None = None


def load_cases(path: Path) -> list[LabeledCase]:
    cases = [LabeledCase.model_validate_json(line) for line in path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("cases must be nonempty and case_id must be unique")
    return cases


def extraction_scores(gold: set[Symptom], predicted: set[Symptom]) -> tuple[int, int, int]:
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


async def evaluate(cases: list[LabeledCase], settings: Settings) -> dict[str, object]:
    agent = IntentAgent(OllamaLLMProvider.from_settings(settings))
    tp = fp = fn = completed = exact = fields_correct = fields_total = 0
    errors: dict[str, int] = {}
    latencies: list[float] = []
    started = time.perf_counter()
    for case in cases:
        t0 = time.perf_counter()
        try:
            result = await agent.extract(case.report)
        except Exception as exc:
            # No raw text, prompts, or exception messages in the results file.
            name = type(exc).__name__
            errors[name] = errors.get(name, 0) + 1
            continue
        latencies.append(round((time.perf_counter() - t0) * 1000, 3))
        completed += 1
        a, b, c = extraction_scores(set(case.symptoms), set(result.symptoms))
        tp += a
        fp += b
        fn += c
        match = (
            set(case.symptoms) == set(result.symptoms)
            and case.severity == result.severity
            and case.duration_minutes == result.duration_minutes
            and case.age == result.age
        )
        exact += int(match)
        fields_correct += sum((
            set(case.symptoms) == set(result.symptoms),
            case.severity == result.severity,
            case.duration_minutes == result.duration_minutes,
            case.age == result.age,
        ))
        fields_total += 4
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "run_id": uuid4().hex,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "task": "structured_symptom_extraction",
        "dataset_type": "fictional_manual_cases",
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "temperature": 0,
        "case_count": len(cases),
        "completed": completed,
        "error_count": len(cases) - completed,
        "error_types": errors,
        "schema_valid_rate": completed / len(cases),
        "exact_match_rate": exact / len(cases),
        "field_accuracy_completed": fields_correct / fields_total if fields_total else None,
        "symptom_micro_precision": precision,
        "symptom_micro_recall": recall,
        "symptom_micro_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "latency_ms_p50": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": (
            sorted(latencies)[min(len(latencies) - 1, max(0, int(len(latencies) * 0.95 + 0.999) - 1))]
            if latencies else None
        ),
        "wall_time_seconds": round(time.perf_counter() - started, 3),
        "note": "Toy fictional cases, not clinical performance or evidence of safe triage.",
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/llm/cases.jsonl"))
    parser.add_argument("--model", help="Override HEALTHFLOW_LLM_MODEL without editing .env")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/llm"))
    args = parser.parse_args()
    settings = Settings(llm_model=args.model) if args.model else Settings()
    cases = load_cases(args.cases)
    result = await evaluate(cases, settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{result['timestamp_utc'][:10]}_{result['run_id']}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Benchmark saved: {output}")
    print(f"model={result['model']} cases={result['case_count']} completed={result['completed']}")


if __name__ == "__main__":
    asyncio.run(main())
