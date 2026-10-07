from typing import Protocol

from app.schemas.care import ServiceType
from app.schemas.facility import Facility, FacilityMatch
from app.tools.geolocation import haversine_km


class FacilityProvider(Protocol):
    async def find_nearby(
        self, service_type: ServiceType, latitude: float, longitude: float, radius_km: float
    ) -> list[FacilityMatch]:
        """Compatible facilities within `radius_km`, nearest first."""
        ...


class MockFacilityProvider:
    """SIMULATED facilities for development. Not a CNES or maps integration."""

    def __init__(self, facilities: list[Facility]) -> None:
        self._facilities = facilities

    async def find_nearby(
        self, service_type: ServiceType, latitude: float, longitude: float, radius_km: float
    ) -> list[FacilityMatch]:
        matches = [
            FacilityMatch(
                facility=facility,
                distance_km=round(
                    haversine_km(latitude, longitude, facility.latitude, facility.longitude), 2
                ),
            )
            for facility in self._facilities
            if facility.service_type is service_type
        ]
        return sorted(
            (m for m in matches if m.distance_km <= radius_km), key=lambda m: m.distance_km
        )


def simulated_facilities() -> list[Facility]:
    """Fictional units around central São Paulo. Names and addresses are invented."""

    def unit(id_: str, name: str, kind: ServiceType, lat: float, lon: float) -> Facility:
        return Facility(
            id=id_,
            name=f"{name} (SIMULADA)",
            service_type=kind,
            address="Endereço fictício para demonstração",
            latitude=lat,
            longitude=lon,
            is_simulated=True,
        )

    return [
        unit("sim-ubs-1", "UBS Exemplo Centro", ServiceType.UBS, -23.5505, -46.6333),
        unit("sim-ubs-2", "UBS Exemplo Vila", ServiceType.UBS, -23.5869, -46.6824),
        unit("sim-upa-1", "UPA Exemplo Norte", ServiceType.UPA, -23.5130, -46.6250),
        unit("sim-upa-2", "UPA Exemplo Sul", ServiceType.UPA, -23.6200, -46.6400),
        unit("sim-er-1", "Pronto-Socorro Exemplo", ServiceType.EMERGENCY_ROOM, -23.5570, -46.6690),
    ]
