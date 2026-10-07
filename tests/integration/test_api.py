from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.dependencies.container import Container
from app.auth.demo_token import DemoTokenAuthenticator
from app.core.config import AppEnv, Settings
from app.core.exceptions import LLMUnavailableError
from app.main import create_app
from app.ml.classifier import RoutingClassifier
from app.repositories.patients import InMemoryPatientRepository, synthetic_demo_patient
from tests.conftest import DEMO_USER_ID, ScriptedLLMProvider, build_harness

TOKEN = "test-token-abcdefghijklmnop"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
VALID_BODY = {"message": "dor no peito e falta de ar", "latitude": -23.55, "longitude": -46.64}
RED_FLAG: dict[str, object] = {
    "symptoms": ["chest_pain", "shortness_of_breath"],
    "severity": "severe",
}


def make_client(llm: ScriptedLLMProvider, classifier: RoutingClassifier | None) -> TestClient:
    settings = Settings(
        app_env=AppEnv.TEST, demo_auth_token=SecretStr(TOKEN), demo_user_id=DEMO_USER_ID
    )
    container = Container(
        harness=build_harness(llm, classifier),
        patients=InMemoryPatientRepository([synthetic_demo_patient(DEMO_USER_ID)]),
        authenticator=DemoTokenAuthenticator(SecretStr(TOKEN), DEMO_USER_ID),
        ml_available=classifier is not None,
    )
    return TestClient(create_app(settings, container), raise_server_exceptions=False)


@pytest.fixture
def client(classifier: RoutingClassifier) -> Iterator[TestClient]:
    with make_client(ScriptedLLMProvider(RED_FLAG), classifier) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ml_model_loaded": True}


def test_routing_returns_emergency_with_request_id(client: TestClient) -> None:
    response = client.post("/api/v1/routing", json=VALID_BODY, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["care_level"] == "EMERGENCY"
    assert body["recommended_service_type"] == "EMERGENCY_ROOM"
    assert body["safety_override"] is True
    assert body["facility"]["is_simulated"] is True
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "diagnosis" not in body


def test_valid_inbound_request_id_is_propagated(client: TestClient) -> None:
    rid = "3f2b8c1e-0000-4000-8000-000000000001"
    response = client.post(
        "/api/v1/routing", json=VALID_BODY, headers={**AUTH, "X-Request-ID": rid}
    )
    assert response.json()["request_id"] == rid


def test_non_uuid_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "evil\ninjected"})
    assert response.headers["X-Request-ID"] != "evil\ninjected"


@pytest.mark.parametrize(
    "body",
    [
        {"message": "", "latitude": 0, "longitude": 0},
        {"message": "dor", "latitude": 91, "longitude": 0},
        {"message": "dor de cabeça", "latitude": 0, "longitude": -181},
        {"message": "x" * 2001, "latitude": 0, "longitude": 0},
        {"message": "dor de cabeça", "latitude": 0, "longitude": 0, "extra": 1},
        {"latitude": 0, "longitude": 0},
    ],
)
def test_invalid_input_returns_422(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/api/v1/routing", json=body, headers=AUTH).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_routing_requires_authentication(client: TestClient, headers: dict[str, str]) -> None:
    response = client.post("/api/v1/routing", json=VALID_BODY, headers=headers)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_patients_me_returns_only_own_record(client: TestClient) -> None:
    response = client.get("/api/v1/patients/me", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["user_id"] == str(DEMO_USER_ID)
    assert client.get("/api/v1/patients/me").status_code == 401


def test_llm_down_returns_safe_503_without_internals(classifier: RoutingClassifier) -> None:
    llm = ScriptedLLMProvider(error=LLMUnavailableError("Connection refused at 10.0.0.5:11434"))
    with make_client(llm, classifier) as client:
        response = client.post(
            "/api/v1/routing", json={**VALID_BODY, "message": "estou com tosse"}, headers=AUTH
        )
    assert response.status_code == 503
    body = response.json()
    assert "10.0.0.5" not in response.text and "Traceback" not in response.text
    assert "192" in body["emergency_hint"]


def test_unexpected_error_returns_generic_500(classifier: RoutingClassifier) -> None:
    llm = ScriptedLLMProvider(error=None)
    llm.payload = {"symptoms": "not-a-list"}  # makes validation explode inside the harness
    with make_client(llm, classifier) as client:
        response = client.post("/api/v1/routing", json=VALID_BODY, headers=AUTH)
    assert response.status_code == 500
    assert response.json()["error"] == "Erro interno."
    assert "ValidationError" not in response.text


def test_demo_auth_is_refused_in_production() -> None:
    with pytest.raises(ValueError, match="production"):
        Settings(app_env=AppEnv.PRODUCTION, demo_auth_token=SecretStr("x"))
