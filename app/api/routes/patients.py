from fastapi import APIRouter, HTTPException, Response

from app.api.dependencies.providers import ContainerDep, PrincipalDep
from app.schemas.patient import PatientRecord

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/me", response_model=PatientRecord)
async def get_my_record(
    principal: PrincipalDep, container: ContainerDep, response: Response, consent: bool = False
) -> PatientRecord:
    if not consent:
        raise HTTPException(status_code=403, detail="Autorize a consulta do prontuário primeiro.")
    response.headers["Cache-Control"] = "no-store"
    # Only the authenticated user's own record is reachable; there is no lookup by id.
    return await container.patients.get_by_user_id(principal.user_id)
