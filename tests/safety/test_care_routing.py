import itertools

import pytest

from app.agents.care_routing_agent import CareRoutingAgent
from app.safety.schemas import SafetyAssessment
from app.schemas.care import CareLevel, ServiceType
from app.schemas.routing import MLPrediction

agent = CareRoutingAgent(low_confidence_threshold=0.55)


def prediction(level: CareLevel, confidence: float = 0.9) -> MLPrediction:
    rest = (1 - confidence) / 2
    probabilities = {lvl: (confidence if lvl is level else rest) for lvl in CareLevel}
    return MLPrediction(
        predicted_class=level, confidence=confidence, probabilities=probabilities, model_version="t"
    )


def safety(minimum: CareLevel | None, rules: list[str] | None = None) -> SafetyAssessment:
    return SafetyAssessment(
        has_red_flag=minimum is CareLevel.EMERGENCY,
        matched_rules=rules or ([] if minimum is None else ["RED_FLAG_001"]),
        suggested_minimum_care_level=minimum,
    )


def test_ml_primary_care_never_downgrades_safety_emergency() -> None:
    decision = agent.decide(safety(CareLevel.EMERGENCY), prediction(CareLevel.PRIMARY_CARE))
    assert decision.care_level is CareLevel.EMERGENCY
    assert decision.service_type is ServiceType.EMERGENCY_ROOM
    assert decision.safety_override is True
    assert "SAFETY_OVERRIDE" in decision.reason_codes


@pytest.mark.parametrize(
    ("floor", "ml_level"),
    list(itertools.product([None, *CareLevel], list(CareLevel))),
)
def test_final_level_respects_safety_and_caps_unvalidated_ml_emergency(
    floor: CareLevel | None, ml_level: CareLevel
) -> None:
    decision = agent.decide(safety(floor), prediction(ml_level))
    expected_ml = (
        CareLevel.URGENT_CARE if ml_level is CareLevel.EMERGENCY else ml_level
    )
    assert decision.care_level.rank >= expected_ml.rank
    if floor is not None:
        assert decision.care_level.rank >= floor.rank


def test_ml_drives_decision_when_no_safety_floor() -> None:
    decision = agent.decide(safety(None), prediction(CareLevel.PRIMARY_CARE))
    assert decision.care_level is CareLevel.PRIMARY_CARE
    assert decision.safety_override is False
    assert decision.reason_codes == ["ML_PREDICTION:PRIMARY_CARE"]


def test_ml_unavailable_falls_back_to_urgent_care() -> None:
    decision = agent.decide(safety(None), None)
    assert decision.care_level is CareLevel.URGENT_CARE
    assert "ML_UNAVAILABLE_CONSERVATIVE_FALLBACK" in decision.reason_codes


def test_red_flag_without_ml_routes_to_emergency() -> None:
    decision = agent.decide(safety(CareLevel.EMERGENCY), None)
    assert decision.care_level is CareLevel.EMERGENCY
    assert decision.safety_override is True


def test_low_confidence_escalates_to_more_severe_of_top_two() -> None:
    ml = MLPrediction(
        predicted_class=CareLevel.PRIMARY_CARE,
        confidence=0.48,
        probabilities={
            CareLevel.PRIMARY_CARE: 0.48,
            CareLevel.URGENT_CARE: 0.42,
            CareLevel.EMERGENCY: 0.10,
        },
        model_version="t",
    )
    decision = agent.decide(safety(None), ml)
    assert decision.care_level is CareLevel.URGENT_CARE
    assert "ML_LOW_CONFIDENCE_ESCALATED" in decision.reason_codes
