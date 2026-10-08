"""Live location lookup using real OpenStreetMap data (not official CNES)."""

import logging

import httpx

from app.core.exceptions import FacilityProviderError
from app.schemas.care import ServiceType
from app.schemas.facility import Facility, FacilityMatch
from app.tools.geolocation import haversine_km

logger = logging.getLogger(__name__)


class OpenStreetMapFacilityProvider:
    def __init__(self, endpoint: str = "https://overpass-api.de/api/interpreter") -> None:
        self.endpoints = (endpoint, "https://overpass.kumi.systems/api/interpreter")

    async def find_nearby(
        self, service_type: ServiceType, latitude: float, longitude: float, radius_km: float
    ) -> list[FacilityMatch]:
        radius = min(int(radius_km * 1000), 50000)
        query = (
            "[out:json][timeout:15];("
            f'nwr(around:{radius},{latitude},{longitude})["amenity"~"hospital|clinic"];'
            f'nwr(around:{radius},{latitude},{longitude})["healthcare"~"hospital|clinic|centre"];'
            f"nwr(around:{radius},{latitude},{longitude})"
            '["name"~"UPA|UBS|Pronto Atendimento|Pronto Socorro|Unidade B[aá]sica",i];'
            ");out center;"
        )
        elements = None
        async with httpx.AsyncClient(timeout=20.0) as client:
            for endpoint in self.endpoints:
                try:
                    response = await client.post(endpoint, data={"data": query})
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload.get("elements"), list):
                        raise ValueError("missing elements")
                    elements = payload["elements"]
                    break
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    logger.warning(
                        "osm_lookup_endpoint_failed",
                        extra={"endpoint": endpoint, "error_type": type(exc).__name__},
                    )
        if elements is None:
            raise FacilityProviderError("all geographic providers unavailable")

        matches = []
        seen = set()
        for item in elements:
            tags = item.get("tags") or {}
            name = str(tags.get("name") or "").strip()
            if not name or not _compatible(name, tags, service_type):
                continue
            point = item.get("center") or item
            lat, lon = point.get("lat"), point.get("lon")
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            distance = haversine_km(latitude, longitude, lat, lon)
            key = (item.get("type"), item.get("id"))
            if distance > radius_km or key in seen:
                continue
            seen.add(key)
            address = (
                ", ".join(
                    str(tags[k])
                    for k in ("addr:street", "addr:housenumber", "addr:city")
                    if tags.get(k)
                )
                or "Endereço não cadastrado no OpenStreetMap"
            )
            matches.append(
                FacilityMatch(
                    facility=Facility(
                        id=f"osm-{key[0]}-{key[1]}",
                        name=name,
                        service_type=service_type,
                        address=address,
                        latitude=lat,
                        longitude=lon,
                        is_simulated=False,
                    ),
                    distance_km=round(distance, 2),
                )
            )
        return sorted(matches, key=lambda item: item.distance_km)


def _compatible(name: str, tags: dict[str, str], service: ServiceType) -> bool:
    value = name.casefold()
    if service is ServiceType.UPA:
        return (
            "upa" in value.split() or "pronto atendimento" in value or "pronto-atendimento" in value
        )
    if service is ServiceType.UBS:
        return (
            "ubs" in value.split()
            or "unidade básica" in value
            or "unidade basica" in value
            or "posto de saúde" in value
        )
    return (
        tags.get("amenity") == "hospital"
        or tags.get("healthcare") == "hospital"
        or "pronto socorro" in value
        or "pronto-socorro" in value
    )
