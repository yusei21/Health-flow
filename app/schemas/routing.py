from pydantic import BaseModel, ConfigDict, Field

from app.schemas.care import CareLevel, ServiceType


class MLPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicted_class: CareLevel
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[CareLevel, float]
    model_version: str


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    care_level: CareLevel
    service_type: ServiceType
    reason_codes: list[str]
    safety_override: bool
    ml_prediction: MLPrediction | None


class RoutingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=3, max_length=2000)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class FacilityResponse(BaseModel):
    name: str
    service_type: ServiceType
    address: str
    latitude: float
    longitude: float
    distance_km: float
    is_simulated: bool


class RoutingResponse(BaseModel):
    request_id: str
    care_level: CareLevel
    recommended_service_type: ServiceType
    facility: FacilityResponse | None
    next_step: str
    emergency_guidance: str | None
    reason_codes: list[str]
    safety_override: bool
    disclaimer: str
