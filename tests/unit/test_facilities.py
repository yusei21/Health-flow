import pytest

from app.agents.navigation_agent import NavigationAgent
from app.schemas.care import ServiceType
from app.tools.facilities import MockFacilityProvider, simulated_facilities
from app.tools.geolocation import haversine_km
from tests.conftest import SAO_PAULO

pytestmark = pytest.mark.anyio


def test_haversine_known_distance_sao_paulo_to_rio() -> None:
    assert haversine_km(-23.5505, -46.6333, -22.9068, -43.1729) == pytest.approx(357, abs=5)


def test_haversine_zero_for_same_point() -> None:
    assert haversine_km(*SAO_PAULO, *SAO_PAULO) == 0


@pytest.mark.parametrize("service_type", list(ServiceType))
async def test_returns_only_compatible_facilities_sorted_by_distance(
    service_type: ServiceType,
) -> None:
    matches = await NavigationAgent(
        MockFacilityProvider(simulated_facilities()), 50
    ).find_facilities(service_type, *SAO_PAULO)
    assert matches
    assert all(m.facility.service_type is service_type for m in matches)
    assert [m.distance_km for m in matches] == sorted(m.distance_km for m in matches)
    assert all(m.facility.is_simulated for m in matches)


async def test_no_facility_outside_radius() -> None:
    manaus = (-3.119, -60.0217)
    matches = await MockFacilityProvider(simulated_facilities()).find_nearby(
        ServiceType.UPA, *manaus, radius_km=25
    )
    assert matches == []
