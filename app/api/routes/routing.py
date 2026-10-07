from fastapi import APIRouter

from app.api.dependencies.providers import ContainerDep, PrincipalDep
from app.core.logging import request_id_var
from app.harness.response import build_routing_response
from app.schemas.routing import RoutingRequest, RoutingResponse

router = APIRouter(prefix="/routing", tags=["routing"])


@router.post("", response_model=RoutingResponse)
async def route_care(
    body: RoutingRequest, principal: PrincipalDep, container: ContainerDep
) -> RoutingResponse:
    state = await container.harness.run(
        request_id=request_id_var.get() or "",
        user_id=principal.user_id,
        message=body.message,
        latitude=body.latitude,
        longitude=body.longitude,
    )
    return build_routing_response(state)
