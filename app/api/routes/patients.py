from fastapi import APIRouter

from app.api.dependencies.providers import ContainerDep, PrincipalDep
from app.schemas.patient import PatientRecord

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/me", response_model=PatientRecord)
async def get_my_record(principal: PrincipalDep, container: ContainerDep) -> PatientRecord:
    # Only the authenticated user's own record is reachable; there is no lookup by id.
    return await container.patients.get_by_user_id(principal.user_id)
