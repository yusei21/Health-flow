from uuid import UUID

from app.context.builder import PatientContextBuilder
from app.core.exceptions import PatientNotFoundError
from app.repositories.patients import PatientRepository
from app.schemas.patient import PatientContext
from app.schemas.symptoms import SymptomExtraction


class PatientContextAgent:
    """Loads the authorized record and reduces it to the minimal routing context."""

    def __init__(self, repository: PatientRepository, builder: PatientContextBuilder) -> None:
        self._repository = repository
        self._builder = builder

    async def build_context(
        self, user_id: UUID, extraction: SymptomExtraction | None
    ) -> PatientContext:
        try:
            record = await self._repository.get_by_user_id(user_id)
        except PatientNotFoundError:
            # Routing must still work for users without a linked record.
            record = None
        return self._builder.build(record, extraction)
