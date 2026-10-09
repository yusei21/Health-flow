"""Do not present unverified OSM results as confirmed SUS services."""

from app.core.exceptions import FacilityProviderError
from app.schemas.care import ServiceType
from app.schemas.facility import FacilityMatch


class UnconfiguredSUSFacilityProvider:
    async def find_nearby(
        self, service_type: ServiceType, latitude: float, longitude: float, radius_km: float
    ) -> list[FacilityMatch]:
        raise FacilityProviderError("official SUS facility source not configured")
