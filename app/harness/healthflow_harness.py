import logging
from uuid import UUID

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.intent_agent import IntentAgent
from app.agents.navigation_agent import NavigationAgent
from app.agents.patient_context_agent import PatientContextAgent
from app.core.exceptions import FacilityProviderError, LLMError, MLInferenceError
from app.core.logging import trace_stage
from app.harness.state import HarnessState, Stage, StageError
from app.ml.inference import RoutingInferenceService
from app.safety.engine import SafetyEngine

logger = logging.getLogger(__name__)


class HealthFlowHarness:
    """Explicit, linear orchestration of the routing workflow.

        extract_user_report -> build_patient_context -> run_safety_assessment
          -> (red flag ? skip ML : run_ml_classifier) -> apply_routing_rules
          -> find_facilities

    Each stage reads/writes `HarnessState` and can be tested on its own. The LLM only
    interprets, the ML only suggests, the safety engine sets floors, and this class
    decides what runs and how failures degrade.
    """

    def __init__(
        self,
        intent_agent: IntentAgent,
        context_agent: PatientContextAgent,
        safety_engine: SafetyEngine,
        inference: RoutingInferenceService,
        routing_agent: CareRoutingAgent,
        navigation_agent: NavigationAgent,
    ) -> None:
        self._intent = intent_agent
        self._context = context_agent
        self._safety = safety_engine
        self._inference = inference
        self._routing = routing_agent
        self._navigation = navigation_agent

    async def run(
        self, request_id: str, user_id: UUID, message: str, latitude: float, longitude: float
    ) -> HarnessState:
        state = HarnessState(
            request_id=request_id,
            user_id=user_id,
            user_message=message,
            latitude=latitude,
            longitude=longitude,
        )
        llm_error = await self.extract_user_report(state)
        await self.build_patient_context(state)
        self.run_safety_assessment(state)
        assert state.safety_assessment is not None  # noqa: S101 - set by previous stage

        if llm_error is not None and not state.safety_assessment.has_red_flag:
            # Without an extraction we can only route when a deterministic red flag
            # fired on the raw text; otherwise surface a controlled error.
            raise llm_error
        if not state.safety_assessment.has_red_flag:
            self.run_ml_classifier(state)
        self.apply_routing_rules(state)
        await self.find_facilities(state)
        return state

    async def extract_user_report(self, state: HarnessState) -> LLMError | None:
        try:
            with trace_stage(logger, Stage.EXTRACT_USER_REPORT):
                state.extracted_symptoms = await self._intent.extract(state.user_message)
        except LLMError as exc:
            state.errors.append(
                StageError(stage=Stage.EXTRACT_USER_REPORT, error_type=type(exc).__name__)
            )
            return exc
        return None

    async def build_patient_context(self, state: HarnessState) -> None:
        with trace_stage(logger, Stage.BUILD_PATIENT_CONTEXT):
            state.patient_context = await self._context.build_context(
                state.user_id, state.extracted_symptoms
            )

    def run_safety_assessment(self, state: HarnessState) -> None:
        with trace_stage(logger, Stage.SAFETY_ASSESSMENT):
            state.safety_assessment = self._safety.assess(
                state.user_message, state.extracted_symptoms, state.patient_context
            )
        logger.info(
            "safety_result",
            extra={
                "red_flag": state.safety_assessment.has_red_flag,
                "matched_rules": state.safety_assessment.matched_rules,
            },
        )

    def run_ml_classifier(self, state: HarnessState) -> None:
        if state.extracted_symptoms is None or state.patient_context is None:
            return
        try:
            with trace_stage(logger, Stage.ML_CLASSIFIER):
                state.ml_prediction = self._inference.predict(
                    state.extracted_symptoms, state.patient_context
                )
        except MLInferenceError as exc:
            # Degrade safely: the routing agent applies a conservative fallback.
            state.errors.append(
                StageError(stage=Stage.ML_CLASSIFIER, error_type=type(exc).__name__)
            )

    def apply_routing_rules(self, state: HarnessState) -> None:
        assert state.safety_assessment is not None  # noqa: S101 - stage order invariant
        with trace_stage(logger, Stage.ROUTING_RULES):
            state.routing_decision = self._routing.decide(
                state.safety_assessment, state.ml_prediction
            )
        logger.info(
            "routing_result",
            extra={
                "care_level": state.routing_decision.care_level,
                "safety_override": state.routing_decision.safety_override,
            },
        )

    async def find_facilities(self, state: HarnessState) -> None:
        assert state.routing_decision is not None  # noqa: S101 - stage order invariant
        try:
            with trace_stage(logger, Stage.FIND_FACILITIES):
                state.facilities = await self._navigation.find_facilities(
                    state.routing_decision.service_type, state.latitude, state.longitude
                )
        except FacilityProviderError as exc:
            # The care level is still valid guidance even without a facility.
            state.errors.append(
                StageError(stage=Stage.FIND_FACILITIES, error_type=type(exc).__name__)
            )
