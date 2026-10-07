from pydantic import BaseModel, ConfigDict, Field

from app.schemas.care import ServiceType


class Facility(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    service_type: ServiceType
    address: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    is_simulated: bool


class FacilityMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    facility: Facility
    distance_km: float
