import asyncio
from unittest.mock import AsyncMock

from app.agents.patient_context_agent import PatientContextAgent
from app.context.builder import PatientContextBuilder
from app.repositories.patients import InMemoryPatientRepository, synthetic_demo_patient
from app.schemas.patient import AgeRange
from app.schemas.symptoms import SymptomExtraction
from tests.conftest import DEMO_USER_ID


def test_denied_consent_never_reads_repository() -> None:
    class DeniedRepository:
        get_by_user_id = AsyncMock(side_effect=AssertionError("unauthorized read"))

    repo = DeniedRepository()
    agent = PatientContextAgent(repo, PatientContextBuilder())
    context = asyncio.run(
        agent.build_context(DEMO_USER_ID, SymptomExtraction(age=30), authorized=False)
    )
    repo.get_by_user_id.assert_not_called()
    assert context.age_range is AgeRange.ADULT
    assert not context.relevant_conditions


def test_authorized_history_changes_context_without_exposing_identity() -> None:
    repo = InMemoryPatientRepository([synthetic_demo_patient(DEMO_USER_ID)])
    context = asyncio.run(
        PatientContextAgent(repo, PatientContextBuilder()).build_context(
            DEMO_USER_ID, SymptomExtraction(), authorized=True
        )
    )
    assert context.age_range is AgeRange.ELDERLY
    assert context.risk_factors
    assert "display_name" not in context.model_dump()
