from app.schemas.care import ServiceType
from app.schemas.facility import FacilityMatch
from app.tools.facilities import FacilityProvider


class NavigationAgent:
    """Finds facilities of the already-decided service type. Never 'any nearest hospital'."""

    def __init__(self, provider: FacilityProvider, radius_km: float) -> None:
        self._provider = provider
        self._radius_km = radius_km

    async def find_facilities(
        self, service_type: ServiceType, latitude: float, longitude: float
    ) -> list[FacilityMatch]:
        return await self._provider.find_nearby(service_type, latitude, longitude, self._radius_km)
