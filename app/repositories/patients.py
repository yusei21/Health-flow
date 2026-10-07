from datetime import date
from typing import Protocol
from uuid import UUID

from app.core.exceptions import PatientNotFoundError
from app.schemas.patient import Condition, Encounter, Medication, PatientRecord, RiskFactor


class PatientRepository(Protocol):
    async def get_by_user_id(self, user_id: UUID) -> PatientRecord: ...


class InMemoryPatientRepository:
    """Explicit non-persistent implementation for development and tests.

    Holds SYNTHETIC records only. Phase F replaces it with a PostgreSQL repository;
    there is no integration with any real SUS system.
    """

    def __init__(self, records: list[PatientRecord]) -> None:
        self._by_user = {record.user_id: record for record in records}

    async def get_by_user_id(self, user_id: UUID) -> PatientRecord:
        try:
            return self._by_user[user_id]
        except KeyError:
            raise PatientNotFoundError(str(user_id)) from None


def synthetic_demo_patient(user_id: UUID) -> PatientRecord:
    """Fictional patient used by the demo login. Not a real person."""
    return PatientRecord(
        patient_id=UUID("0b9c3f52-8f7e-4d1a-b6a4-2e5d9c7f1a30"),
        user_id=user_id,
        display_name="Paciente Demonstração (fictício)",
        age=67,
        allergies=["dipirona"],
        conditions=[
            Condition(name="hipertensão arterial", risk_factor=RiskFactor.CARDIOVASCULAR_DISEASE),
            Condition(name="diabetes tipo 2", risk_factor=RiskFactor.DIABETES),
        ],
        active_medications=[
            Medication(name="losartana"),
            Medication(name="varfarina", is_anticoagulant=True),
        ],
        previous_encounters=[
            Encounter(occurred_on=date(2026, 3, 14), service="UBS", summary="Consulta de rotina."),
        ],
    )
