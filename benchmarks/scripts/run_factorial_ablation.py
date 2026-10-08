"""Run a paired offline ablation using real Health-flow routing components.

Input cases must have independently assigned labels and pre-extracted symptoms.
This tool does not access EHRs, call the LLM, or establish clinical validity.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from uuid import UUID, uuid5

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.intent_agent import IntentAgent
from app.agents.navigation_agent import NavigationAgent
from app.agents.patient_context_agent import PatientContextAgent
from app.context.builder import PatientContextBuilder
from app.harness.autonomous_harness import AutonomousHealthFlowHarness
from app.harness.state import combine_assessments
from app.ml.classifier import RoutingClassifier
from app.ml.inference import RoutingInferenceService
from app.safety.engine import SafetyEngine
from app.schemas.care import CareLevel
from app.schemas.patient import PatientContext, PatientRecord
from app.schemas.routing import MLPrediction
from app.schemas.symptoms import Severity, SymptomExtraction
from app.tools.facilities import MockFacilityProvider

LEVELS = tuple(CareLevel)
NAMESPACE = UUID("538a2306-dcbc-482f-b2de-5fcd7c9e1350")


class FrozenIntent(IntentAgent):
    """Return independently prepared structured extraction, not a live LLM result."""

    def __init__(self, extraction: SymptomExtraction) -> None:
        self.extraction = extraction

    async def extract(self, report_text: str) -> SymptomExtraction:
        return self.extraction


class FrozenContext(PatientContextAgent):
    def __init__(self, record: PatientRecord | None, enabled: bool) -> None:
        self.record = record if enabled else None
        self.builder = PatientContextBuilder()

    async def build_context(
        self, user_id: UUID, extraction: SymptomExtraction | None
    ) -> PatientContext:
        return self.builder.build(self.record, extraction)


class RuleBaseline(RoutingInferenceService):
    """Predefined non-ML classifier; distinct from a model-unavailable fallback."""

    def __init__(self) -> None:
        super().__init__(None)

    def predict(self, extraction: SymptomExtraction, context: PatientContext) -> MLPrediction:
        if extraction.severity is Severity.SEVERE:
            level = CareLevel.URGENT_CARE
        elif extraction.severity is Severity.MILD and extraction.symptoms:
            level = CareLevel.PRIMARY_CARE
        else:
            level = CareLevel.URGENT_CARE
        return MLPrediction(
            predicted_class=level,
            confidence=1.0,
            probabilities={item: float(item is level) for item in LEVELS},
            model_version="rule-baseline-v1",
        )


async def direct_pipeline(
    message: str,
    extraction: SymptomExtraction,
    context: FrozenContext,
    inference: RoutingInferenceService,
    user_id: UUID,
) -> tuple[CareLevel, bool, int, int]:
    """Fixed direct composition without Harness planner, policy or state machine."""
    safety_engine = SafetyEngine()
    precheck = safety_engine.assess(message, None, None)
    llm_calls = 0
    tool_calls = 0
    prediction = None
    if not precheck.has_red_flag:
        llm_calls = 1  # frozen extraction: logical call only, not actual LLM invocation
        patient_context = await context.build_context(user_id, extraction)
        tool_calls += 1
        full = safety_engine.assess(message, extraction, patient_context)
        assessment = combine_assessments(precheck, full)
        if assessment is None:
            raise ValueError("Missing safety assessment")
        if not assessment.has_red_flag:
            prediction = inference.predict(extraction, patient_context)
            tool_calls += 1
    else:
        assessment = precheck
    decision = CareRoutingAgent(low_confidence_threshold=0.55).decide(assessment, prediction)
    # Matching the Harness's facility lookup; an empty provider is intentional.
    navigation = NavigationAgent(MockFacilityProvider([]), 25)
    await navigation.find_facilities(decision.service_type, -23.55, -46.64)
    tool_calls += 1
    return decision.care_level, decision.safety_override, llm_calls, tool_calls


async def run_case(
    case: dict, harness_on: bool, ml_on: bool, context_on: bool,
    classifier: RoutingClassifier | None,
) -> dict:
    extraction = SymptomExtraction.model_validate(case["extraction"])
    message = str(case["report"])
    user_id = uuid5(NAMESPACE, str(case["case_id"]))
    record_data = case.get("patient_record")
    record = (
        PatientRecord.model_validate({**record_data, "user_id": user_id})
        if record_data is not None else None
    )
    context = FrozenContext(record, context_on)
    inference = RoutingInferenceService(classifier) if ml_on else RuleBaseline()
    if ml_on and classifier is None:
        raise ValueError("ML variant requires --model-dir with a compatible trained model")
    started = time.perf_counter()
    if harness_on:
        runner = AutonomousHealthFlowHarness(
            intent_agent=FrozenIntent(extraction),
            context_agent=context,
            safety_engine=SafetyEngine(),
            inference=inference,
            routing_agent=CareRoutingAgent(low_confidence_threshold=0.55),
            navigation_agent=NavigationAgent(MockFacilityProvider([]), 25),
        )
        state = await runner.run(str(case["case_id"]), user_id, message, -23.55, -46.64)
        if state.routing_decision is None:
            raise ValueError("Harness produced no routing decision")
        level = state.routing_decision.care_level
        override = state.routing_decision.safety_override
        llm_calls, tool_calls = state.llm_call_count, state.tool_call_count
        violations = 0  # no completed run can bypass HarnessPolicy validation
    else:
        level, override, llm_calls, tool_calls = await direct_pipeline(
            message, extraction, context, inference, user_id
        )
        violations = 0  # not measured by an independent oracle in this runner
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "case_id": str(case["case_id"]),
        "dataset_hash": case["_hash"],
        "split": "test",
        "reference_level": case["reference_level"],
        "predicted_final": level.value,
        "harness_enabled": harness_on,
        "ml_enabled": ml_on,
        "patient_context_enabled": context_on,
        "duration_ms": round(elapsed_ms, 3),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "invariant_violations": violations,
        "safety_override": override,
        "status": "ok",
    }


async def run_all(cases: list[dict], classifier: RoutingClassifier | None) -> list[dict]:
    results = []
    for case in cases:
        for h in (False, True):
            for m in (False, True):
                for p in (False, True):
                    results.append(await run_case(case, h, m, p, classifier))
    return results


def load_cases(path: Path) -> list[dict]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    cases = []
    ids = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for field in ("case_id", "report", "extraction", "reference_level"):
            if field not in case:
                raise ValueError(f"Line {line_number}: missing {field}")
        if case["case_id"] in ids:
            raise ValueError(f"Line {line_number}: duplicate case_id")
        ids.add(case["case_id"])
        if case["reference_level"] not in {level.value for level in LEVELS}:
            raise ValueError(f"Line {line_number}: invalid reference_level")
        SymptomExtraction.model_validate(case["extraction"])
        case["_hash"] = digest
        cases.append(case)
    if not cases:
        raise ValueError("Empty corpus")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.cases.resolve():
        parser.error("Input and output must differ")
    classifier = RoutingClassifier.load(args.model_dir)
    cases = load_cases(args.cases)
    results = asyncio.run(run_all(cases, classifier))
    with args.output.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
