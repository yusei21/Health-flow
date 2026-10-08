"""Controlled failure-injection comparison: Harness versus unguarded direct sequence.

Tests software recovery and policy enforcement, not clinical safety or reliability rates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.navigation_agent import NavigationAgent
from app.core.exceptions import (
    FacilityProviderError,
    HarnessError,
    LLMUnavailableError,
    MLInferenceError,
)
from app.harness.actions import HarnessAction, PlannedAction
from app.harness.autonomous_harness import AutonomousHealthFlowHarness
from app.harness.state import HarnessState
from app.ml.inference import RoutingInferenceService
from app.safety.engine import SafetyEngine
from app.schemas.care import ServiceType
from app.schemas.facility import FacilityMatch
from app.schemas.patient import PatientContext
from app.schemas.routing import MLPrediction
from app.schemas.symptoms import SymptomExtraction
from app.tools.facilities import MockFacilityProvider
from benchmarks.scripts.run_factorial_ablation import FrozenContext, FrozenIntent, RuleBaseline

USER = UUID("7d6f0f3e-2b8a-4c1e-9a52-3f4b8c2d1e90")
MESSAGE = "estou com tosse leve"
EXTRACTION = SymptomExtraction.model_validate({"symptoms": ["cough"], "severity": "mild"})


class FailingInference(RoutingInferenceService):
    def __init__(self) -> None:
        super().__init__(None)

    def predict(self, extraction: SymptomExtraction, context: PatientContext) -> MLPrediction:
        raise MLInferenceError("injected ml outage")


class FailingNavigation(NavigationAgent):
    def __init__(self) -> None:
        super().__init__(MockFacilityProvider([]), 25)

    async def find_facilities(
        self, service_type: ServiceType, latitude: float, longitude: float
    ) -> list[FacilityMatch]:
        raise FacilityProviderError("injected facility outage")


class FailingIntent(FrozenIntent):
    async def extract(self, report_text: str) -> SymptomExtraction:
        raise LLMUnavailableError("injected extraction outage")


class InvalidPlanner:
    async def next_action(self, state: HarnessState) -> PlannedAction:
        return PlannedAction(HarnessAction.APPLY_ROUTING, "INJECTED_SKIP_PRECHECK")


def components(
    failure: str,
) -> tuple[FrozenIntent, RoutingInferenceService, NavigationAgent, FrozenContext]:
    intent = FailingIntent(EXTRACTION) if failure == "llm" else FrozenIntent(EXTRACTION)
    inference = FailingInference() if failure == "ml" else RuleBaseline()
    nav = (
        FailingNavigation()
        if failure == "facilities"
        else NavigationAgent(MockFacilityProvider([]), 25)
    )
    context = FrozenContext(None, True)
    return intent, inference, nav, context


async def harness_trial(failure: str) -> dict[str, object]:
    intent, inference, nav, context = components(failure)
    runner = AutonomousHealthFlowHarness(
        intent_agent=intent,
        context_agent=context,
        safety_engine=SafetyEngine(),
        inference=inference,
        routing_agent=CareRoutingAgent(low_confidence_threshold=0.55),
        navigation_agent=nav,
        planner=InvalidPlanner() if failure == "policy" else None,
    )
    try:
        state = await runner.run("injected", USER, MESSAGE, -23.55, -46.64)
    except (FacilityProviderError, HarnessError, LLMUnavailableError, MLInferenceError) as exc:
        return {"status": "raised", "error_type": type(exc).__name__, "decision": None}
    return {
        "status": "completed",
        "decision": state.routing_decision.care_level.value if state.routing_decision else None,
        "handled_errors": [error.error_type for error in state.errors],
        "action_count": state.step_count,
    }


async def direct_trial(failure: str) -> dict[str, object]:
    intent, inference, nav, context_agent = components(failure)
    try:
        safety = SafetyEngine()
        safety.assess(MESSAGE, None, None)
        # No planner or policy exists in the direct branch; 'policy' is an invalid
        # ordering attempt which the direct sequence refuses to simulate as safe.
        if failure == "policy":
            return {"status": "not_applicable", "decision": None}
        extraction = await intent.extract(MESSAGE)
        context = await context_agent.build_context(USER, extraction)
        full = safety.assess(MESSAGE, extraction, context)
        prediction = inference.predict(extraction, context) if not full.has_red_flag else None
        routing = CareRoutingAgent(low_confidence_threshold=0.55).decide(full, prediction)
        await nav.find_facilities(routing.service_type, -23.55, -46.64)
    except (FacilityProviderError, HarnessError, LLMUnavailableError, MLInferenceError) as exc:
        return {"status": "raised", "error_type": type(exc).__name__, "decision": None}
    return {"status": "completed", "decision": routing.care_level.value}


async def evaluate() -> dict[str, Any]:
    trials = []
    for failure in ("none", "ml", "facilities", "llm", "policy"):
        trials.append(
            {
                "injected_failure": failure,
                "harness": await harness_trial(failure),
                "direct": await direct_trial(failure),
            }
        )
    return {
        "status": "deterministic_failure_injection_diagnostic",
        "trials": trials,
        "limitations": [
            "Scripted failures, not estimates of failure frequency",
            "Direct pipeline has no recovery wrappers by design: compare exception propagation",
            "Policy violation has no equivalent planner action in direct sequence",
            "Neither live Ollama nor real facility APIs are used",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = asyncio.run(evaluate())
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sys.stdout.write("Failure injection completed: 5 scenarios (including baseline).\n")


if __name__ == "__main__":
    main()
