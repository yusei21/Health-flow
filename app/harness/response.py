from app.harness.state import HarnessState
from app.schemas.care import CareLevel
from app.schemas.routing import FacilityResponse, RoutingResponse

DISCLAIMER = (
    "Health-flow fornece orientação de navegação em saúde e não substitui avaliação "
    "profissional. Em caso de emergência, ligue 192 (SAMU)."
)
EMERGENCY_GUIDANCE = (
    "Possível situação de emergência. Ligue imediatamente para o SAMU 192. "
    "Este sistema NÃO aciona ambulância automaticamente."
)
_NEXT_STEP = {
    CareLevel.PRIMARY_CARE: "Procure uma Unidade Básica de Saúde (UBS) para avaliação.",
    CareLevel.URGENT_CARE: "Procure atendimento de urgência (UPA) o quanto antes.",
    CareLevel.EMERGENCY: "Ligue 192 (SAMU) ou dirija-se imediatamente a um pronto-socorro.",
}
_NO_FACILITY = " Nenhuma unidade compatível foi encontrada perto da localização informada."


def build_routing_response(state: HarnessState) -> RoutingResponse:
    decision = state.routing_decision
    if decision is None:
        raise ValueError("routing decision missing; harness did not complete")
    nearest = state.facilities[0] if state.facilities else None
    facility = (
        FacilityResponse(
            name=nearest.facility.name,
            service_type=nearest.facility.service_type,
            address=nearest.facility.address,
            latitude=nearest.facility.latitude,
            longitude=nearest.facility.longitude,
            distance_km=nearest.distance_km,
            is_simulated=nearest.facility.is_simulated,
        )
        if nearest
        else None
    )
    is_emergency = decision.care_level is CareLevel.EMERGENCY
    return RoutingResponse(
        request_id=state.request_id,
        care_level=decision.care_level,
        recommended_service_type=decision.service_type,
        facility=facility,
        next_step=_NEXT_STEP[decision.care_level] + ("" if facility else _NO_FACILITY),
        emergency_guidance=EMERGENCY_GUIDANCE if is_emergency else None,
        reason_codes=decision.reason_codes,
        safety_override=decision.safety_override,
        disclaimer=DISCLAIMER,
    )
