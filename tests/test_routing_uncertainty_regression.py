"""Regression checks for conservative routing and text negations."""
from app.agents.care_routing_agent import CareRoutingAgent
from app.safety.engine import SafetyEngine
from app.safety.schemas import SafetyAssessment
from app.schemas.care import CareLevel
from app.schemas.routing import MLPrediction


def test_unvalidated_ml_cannot_trigger_emergency_alone() -> None:
    safety = SafetyAssessment(
        has_red_flag=False, matched_rules=[], suggested_minimum_care_level=None
    )
    prediction = MLPrediction(
        predicted_class=CareLevel.EMERGENCY, confidence=0.95,
        probabilities={
            CareLevel.PRIMARY_CARE: 0.01,
            CareLevel.URGENT_CARE: 0.04,
            CareLevel.EMERGENCY: 0.95,
        },
        model_version="test",
    )
    result = CareRoutingAgent(0.55).decide(safety, prediction)
    assert result.care_level is CareLevel.URGENT_CARE


def test_explicit_emergency_safety_rule_preserved() -> None:
    safety = SafetyAssessment(
        has_red_flag=True, matched_rules=["RED_FLAG_002"],
        suggested_minimum_care_level=CareLevel.EMERGENCY,
    )
    result = CareRoutingAgent(0.55).decide(safety, None)
    assert result.care_level is CareLevel.EMERGENCY


def test_simple_negation_is_not_a_red_flag() -> None:
    assessment = SafetyEngine().assess("nao desmaiei", None, None)
    assert not assessment.has_red_flag


def test_affirmed_fainting_is_still_a_red_flag() -> None:
    assessment = SafetyEngine().assess("desmaiei agora", None, None)
    assert assessment.has_red_flag
