from app.harness.actions import HarnessAction
from app.harness.follow_up import follow_up_questions
from app.harness.state import HarnessState
from app.harness.sus_context import NEXT_STEP, SUS_CONTEXT_VERSION
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
    facility_lookup_failed = any(
        error.stage is HarnessAction.SEARCH_FACILITIES for error in state.errors
    )
    no_facility_note = (
        " A busca de unidades está temporariamente indisponível; "
        "não significa que não existam unidades próximas."
        if facility_lookup_failed
        else _NO_FACILITY
    )
    extraction = state.extracted_symptoms
    uncertain = (
        not state.has_red_flag
        and not (state.effective_safety and state.effective_safety.matched_rules)
        and extraction is not None
        and extraction.severity.value == "unknown"
        and extraction.duration_minutes is None
    )
    questions = follow_up_questions(state.user_message, extraction) if uncertain else []
    is_emergency = decision.care_level is CareLevel.EMERGENCY and not uncertain
    return RoutingResponse(
        request_id=state.request_id,
        care_level=decision.care_level,
        recommended_service_type=decision.service_type,
        facility=facility,
        next_step=(
            "Não é possível determinar a gravidade apenas com esse relato. "
            "Se houver sinais de risco imediato, acione o SAMU 192. "
            "Caso contrário, procure avaliação profissional conforme os sintomas."
            if uncertain
            else NEXT_STEP[decision.care_level]
        )
        + ("" if facility else no_facility_note),
        emergency_guidance=EMERGENCY_GUIDANCE if is_emergency else None,
        reason_codes=decision.reason_codes,
        safety_override=decision.safety_override,
        disclaimer=DISCLAIMER,
        needs_more_information=uncertain,
        follow_up_questions=questions,
        sus_context_version=SUS_CONTEXT_VERSION,
        patient_record_authorized=state.use_patient_record,
    )
