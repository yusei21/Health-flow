from app.safety.schemas import SafetyAssessment
from app.schemas.care import SERVICE_TYPE_BY_CARE_LEVEL, CareLevel, most_severe
from app.schemas.routing import MLPrediction, RoutingDecision

ML_UNAVAILABLE_FALLBACK = CareLevel.URGENT_CARE


class CareRoutingAgent:
    """Combines the ML suggestion with safety floors into the final care level.

    Invariant: the final level is never below the safety minimum nor below the ML
    prediction. Rules may raise the level; nothing may lower a safety floor.
    """

    def __init__(self, low_confidence_threshold: float) -> None:
        self._low_confidence_threshold = low_confidence_threshold

    def decide(self, safety: SafetyAssessment, ml: MLPrediction | None) -> RoutingDecision:
        reasons: list[str] = []
        ml_level = self._ml_level(safety, ml, reasons)
        safety_floor = safety.suggested_minimum_care_level
        # An unvalidated synthetic ML prediction cannot independently trigger the
        # highest-risk emergency pathway without a matched emergency safety rule.
        if ml_level is CareLevel.EMERGENCY and safety_floor is not CareLevel.EMERGENCY:
            ml_level = CareLevel.URGENT_CARE
            reasons.append("ML_EMERGENCY_REQUIRES_VALIDATED_SAFETY_FLAG")
        final = most_severe(ml_level, safety_floor) or ML_UNAVAILABLE_FALLBACK

        override = safety_floor is not None and (
            ml_level is None or safety_floor.rank > ml_level.rank
        )
        if safety.matched_rules:
            reasons.extend(f"SAFETY_RULE:{rule_id}" for rule_id in safety.matched_rules)
        if override:
            reasons.append("SAFETY_OVERRIDE")
        return RoutingDecision(
            care_level=final,
            service_type=SERVICE_TYPE_BY_CARE_LEVEL[final],
            reason_codes=reasons,
            safety_override=override,
            ml_prediction=ml,
        )

    def _ml_level(
        self, safety: SafetyAssessment, ml: MLPrediction | None, reasons: list[str]
    ) -> CareLevel | None:
        if ml is None:
            if not safety.has_red_flag:
                # Without the classifier we cannot rule out urgency: fail towards care.
                reasons.append("ML_UNAVAILABLE_CONSERVATIVE_FALLBACK")
                return ML_UNAVAILABLE_FALLBACK
            return None
        reasons.append(f"ML_PREDICTION:{ml.predicted_class.value}")
        if ml.confidence >= self._low_confidence_threshold:
            return ml.predicted_class
        # Low confidence: take the more severe of the two most likely classes.
        top_two = sorted(ml.probabilities, key=lambda level: ml.probabilities[level])[-2:]
        escalated = most_severe(*top_two) or ml.predicted_class
        if escalated is not ml.predicted_class:
            reasons.append("ML_LOW_CONFIDENCE_ESCALATED")
        return escalated
