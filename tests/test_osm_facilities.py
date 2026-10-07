"""Basic categorization checks for live geographic provider (offline)."""
from app.schemas.care import ServiceType
from app.tools.osm_facilities import _compatible


def test_upa_name_matches_only_appropriate_service() -> None:
    assert _compatible("UPA Vila Nova", {}, ServiceType.UPA)
    assert not _compatible("UBS Vila Nova", {}, ServiceType.UPA)


def test_primary_care_name_matches_ubs() -> None:
    assert _compatible("UBS Central", {}, ServiceType.UBS)
    assert not _compatible("UPA Central", {}, ServiceType.UBS)


def test_hospital_for_emergencies() -> None:
    assert _compatible("Hospital Municipal", {"amenity": "hospital"}, ServiceType.EMERGENCY_ROOM)
