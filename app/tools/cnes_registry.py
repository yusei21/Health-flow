"""Import and query an official CNES CSV export with an explicit field mapping.

CNES describes registered establishments, NOT real-time availability or a
guarantee that a hospital operates an emergency room.
"""

import csv
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from app.core.exceptions import FacilityProviderError
from app.schemas.care import ServiceType
from app.schemas.facility import Facility, FacilityMatch
from app.tools.geolocation import haversine_km

# CNES establishment type codes. 73 is generic pronto atendimento, not
# necessarily a certified UPA. 05/07 indicate hospitals, not verified EDs.
TYPES = {
    "01": ServiceType.UBS, "02": ServiceType.UBS,
    "05": ServiceType.EMERGENCY_ROOM, "07": ServiceType.EMERGENCY_ROOM,
    "20": ServiceType.EMERGENCY_ROOM, "21": ServiceType.EMERGENCY_ROOM,
    "73": ServiceType.UPA,
}
FIELDS = (
    "cnes", "name", "type_code", "street", "number", "district",
    "city", "state", "latitude", "longitude", "sus",
)


def import_cnes_csv(
    csv_path: Path, database: Path, columns: dict[str, str],
    reference_date: date, source_url: str, delimiter: str = ";",
) -> int:
    """Atomically create an offline snapshot. Never assumes undocumented column names."""
    if set(columns) != set(FIELDS):
        raise ValueError("explicit mapping required for all CNES fields")
    if not source_url.startswith("https://"):
        raise ValueError("HTTPS source URL required")
    if delimiter not in (";", ",", "\t"):
        raise ValueError("unsupported delimiter")
    database.parent.mkdir(parents=True, exist_ok=True)
    staging = database.with_suffix(database.suffix + ".new")
    staging.unlink(missing_ok=True)
    try:
        with sqlite3.connect(staging) as db:
            db.execute(
                "CREATE TABLE facilities (cnes TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "type_code TEXT NOT NULL, service_type TEXT NOT NULL, address TEXT NOT NULL, "
                "latitude REAL NOT NULL, longitude REAL NOT NULL, "
                "reference_date TEXT NOT NULL, source_url TEXT NOT NULL, imported_at TEXT NOT NULL)"
            )
            db.execute("CREATE INDEX facilities_geo ON facilities(latitude, longitude)")
            count = 0
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter=delimiter)
                missing = set(columns.values()) - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"CSV missing mapped columns: {sorted(missing)}")
                for row in reader:
                    cnes = (row[columns["cnes"]] or "").strip()
                    code = (row[columns["type_code"]] or "").strip().zfill(2)
                    sus = (row[columns["sus"]] or "").strip().upper()
                    if not (cnes.isdecimal() and len(cnes) == 7
                            and code in TYPES and sus in {"1", "S", "SIM", "TRUE"}):
                        continue
                    name = (row[columns["name"]] or "").strip()
                    try:
                        lat = float((row[columns["latitude"]] or "").replace(",", "."))
                        lon = float((row[columns["longitude"]] or "").replace(",", "."))
                    except ValueError:
                        continue
                    if not name or not (-34 <= lat <= 6 and -74 <= lon <= -28):
                        continue
                    address = ", ".join(
                        value for key in ("street", "number", "district", "city", "state")
                        if (value := (row[columns[key]] or "").strip())
                    )
                    if not address:
                        continue
                    db.execute(
                        "INSERT OR REPLACE INTO facilities VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (cnes, name, code, TYPES[code].value, address, lat, lon,
                         reference_date.isoformat(), source_url,
                         datetime.now(timezone.utc).isoformat()),
                    )
                    count += 1
            if not count:
                raise ValueError("no eligible geocoded SUS facilities; existing snapshot preserved")
        staging.replace(database)
        return count
    finally:
        staging.unlink(missing_ok=True)


class CNESFacilityProvider:
    """Offline registry. Do not imply open hours, vacancies, or ED availability."""

    def __init__(self, database: Path) -> None:
        self.database = database

    async def find_nearby(
        self, service_type: ServiceType, latitude: float, longitude: float, radius_km: float
    ) -> list[FacilityMatch]:
        try:
            with sqlite3.connect(f"file:{self.database}?mode=ro", uri=True) as db:
                rows = db.execute(
                    "SELECT cnes, name, address, latitude, longitude FROM facilities "
                    "WHERE service_type = ?", (service_type.value,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise FacilityProviderError("CNES snapshot unavailable") from exc
        results = []
        for cnes, name, address, lat, lon in rows:
            distance = haversine_km(latitude, longitude, lat, lon)
            if distance <= radius_km:
                results.append(
                    FacilityMatch(
                        facility=Facility(
                            id=f"cnes-{cnes}", name=name, service_type=service_type,
                            address=address, latitude=lat, longitude=lon,
                            is_simulated=False,
                        ),
                        distance_km=round(distance, 2),
                    )
                )
        return sorted(results, key=lambda match: match.distance_km)
