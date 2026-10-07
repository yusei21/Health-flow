"""Composition root: the only place that chooses concrete implementations."""

import logging
from dataclasses import dataclass

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.intent_agent import IntentAgent
from app.agents.navigation_agent import NavigationAgent
from app.agents.patient_context_agent import PatientContextAgent
from app.auth.demo_token import DemoTokenAuthenticator
from app.context.builder import PatientContextBuilder
from app.core.config import Settings
from app.core.exceptions import MLModelUnavailableError
from app.harness.autonomous_harness import AutonomousHealthFlowHarness
from app.llm.ollama_provider import OllamaLLMProvider
from app.ml.classifier import RoutingClassifier
from app.ml.feature_builders import HealthFlowSymptomFeatureBuilder
from app.ml.inference import RoutingInferenceService
from app.repositories.patients import (
    InMemoryPatientRepository,
    PatientRepository,
    synthetic_demo_patient,
)
from app.safety.engine import SafetyEngine
from app.tools.osm_facilities import OpenStreetMapFacilityProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Container:
    harness: AutonomousHealthFlowHarness
    patients: PatientRepository
    authenticator: DemoTokenAuthenticator
    ml_available: bool


def load_classifier(settings: Settings) -> RoutingClassifier | None:
    try:
        # The API computes symptom features only; a vitals-based model cannot be served.
        classifier = RoutingClassifier.load(
            settings.ml_model_dir, expected_feature_set=HealthFlowSymptomFeatureBuilder.feature_set
        )
    except MLModelUnavailableError as exc:
        logger.warning("ml_model_unavailable", extra={"reason": str(exc)})
        return None
    logger.info(
        "ml_model_loaded",
        extra={
            "model_type": classifier.metadata.model_type,
            "version": classifier.metadata.model_version,
        },
    )
    return classifier


def build_container(settings: Settings) -> Container:
    records = [synthetic_demo_patient(settings.demo_user_id)] if settings.demo_user_id else []
    patients = InMemoryPatientRepository(records)
    inference = RoutingInferenceService(load_classifier(settings))
    harness = AutonomousHealthFlowHarness(
        intent_agent=IntentAgent(OllamaLLMProvider.from_settings(settings)),
        context_agent=PatientContextAgent(patients, PatientContextBuilder()),
        safety_engine=SafetyEngine(),
        inference=inference,
        routing_agent=CareRoutingAgent(settings.ml_low_confidence_threshold),
        navigation_agent=NavigationAgent(
            OpenStreetMapFacilityProvider(), settings.facility_search_radius_km
        ),
    )
    return Container(
        harness=harness,
        patients=patients,
        authenticator=DemoTokenAuthenticator(settings.demo_auth_token, settings.demo_user_id),
        ml_available=inference.available,
    )
