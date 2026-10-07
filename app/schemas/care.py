"""Central care-navigation vocabulary. Use these enums instead of raw strings."""

from enum import StrEnum


class CareLevel(StrEnum):
    PRIMARY_CARE = "PRIMARY_CARE"
    URGENT_CARE = "URGENT_CARE"
    EMERGENCY = "EMERGENCY"

    @property
    def rank(self) -> int:
        return _CARE_LEVEL_RANK[self]


_CARE_LEVEL_RANK = {
    CareLevel.PRIMARY_CARE: 0,
    CareLevel.URGENT_CARE: 1,
    CareLevel.EMERGENCY: 2,
}


def most_severe(*levels: CareLevel | None) -> CareLevel | None:
    present = [level for level in levels if level is not None]
    return max(present, key=lambda level: level.rank) if present else None


class ServiceType(StrEnum):
    UBS = "UBS"
    UPA = "UPA"
    EMERGENCY_ROOM = "EMERGENCY_ROOM"


SERVICE_TYPE_BY_CARE_LEVEL: dict[CareLevel, ServiceType] = {
    CareLevel.PRIMARY_CARE: ServiceType.UBS,
    CareLevel.URGENT_CARE: ServiceType.UPA,
    CareLevel.EMERGENCY: ServiceType.EMERGENCY_ROOM,
}
