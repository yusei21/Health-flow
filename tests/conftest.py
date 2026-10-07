from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel

from app.agents.care_routing_agent import CareRoutingAgent
from app.agents.intent_agent import IntentAgent
from app.agents.navigation_agent import NavigationAgent
from app.agents.patient_context_agent import PatientContextAgent
from app.context.builder import PatientContextBuilder
from app.core.exceptions import LLMError
from app.harness.healthflow_harness import HealthFlowHarness
from app.llm.schemas import ChatMessage
from app.ml.classifier import RoutingClassifier
from app.ml.data.synthetic import generate_examples, save_dataset
from app.ml.experiments import EXPERIMENTS
from app.ml.inference import RoutingInferenceService
from app.ml.training.train import run_experiment
from app.repositories.patients import InMemoryPatientRepository, synthetic_demo_patient
from app.safety.engine import SafetyEngine
from app.tools.facilities import MockFacilityProvider, simulated_facilities

DEMO_USER_ID = UUID("7d6f0f3e-2b8a-4c1e-9a52-3f4b8c2d1e90")
SAO_PAULO = (-23.55, -46.64)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedLLMProvider:
    """Test double: returns a scripted JSON payload or raises a scripted error."""

    def __init__(self, payload: dict[str, object] | None = None, error: LLMError | None = None):
        self.payload = payload or {}
        self.error = error
        self.calls: list[Sequence[ChatMessage]] = []

    async def generate_structured[T: BaseModel](
        self, messages: Sequence[ChatMessage], output_model: type[T]
    ) -> T:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return output_model.model_validate(self.payload)


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ml")
    dataset = root / "dataset.csv"
    save_dataset(generate_examples(1500, seed=7), dataset)
    run_experiment(
        EXPERIMENTS["synthetic_baseline"], dataset, root / "model", benchmark_dir=root / "bench"
    )
    return root / "model"


@pytest.fixture(scope="session")
def classifier(trained_model_dir: Path) -> RoutingClassifier:
    return RoutingClassifier.load(trained_model_dir)


def build_harness(
    llm: ScriptedLLMProvider,
    classifier: RoutingClassifier | None,
    facilities: MockFacilityProvider | None = None,
    radius_km: float = 25,
) -> HealthFlowHarness:
    repository = InMemoryPatientRepository([synthetic_demo_patient(DEMO_USER_ID)])
    return HealthFlowHarness(
        intent_agent=IntentAgent(llm),
        context_agent=PatientContextAgent(repository, PatientContextBuilder()),
        safety_engine=SafetyEngine(),
        inference=RoutingInferenceService(classifier),
        routing_agent=CareRoutingAgent(low_confidence_threshold=0.55),
        navigation_agent=NavigationAgent(
            facilities or MockFacilityProvider(simulated_facilities()), radius_km
        ),
    )
