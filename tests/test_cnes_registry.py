"""Offline CNES ingestion tests. Fixture rows are synthetic, not actual facilities."""

from datetime import date
from pathlib import Path

import pytest

from app.schemas.care import ServiceType
from app.tools.cnes_registry import CNESFacilityProvider, FIELDS, import_cnes_csv

COLUMNS = {key: key for key in FIELDS}
SOURCE = "https://dadosabertos.saude.gov.br/dataset/cnes-cadastro-nacional-de-estabelecimentos-de-saude"


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        ";".join(FIELDS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )


def row(cnes: str, kind: str, sus: str, lat: str = "-23.55") -> str:
    return ";".join(
        [cnes, "Unidade de teste", kind, "Rua fictícia", "1", "Centro",
         "São Paulo", "SP", lat, "-46.63", sus]
    )


@pytest.mark.anyio
async def test_import_filters_non_sus_and_invalid_locations(tmp_path: Path) -> None:
    source = tmp_path / "cnes.csv"
    db = tmp_path / "facilities.sqlite"
    write_csv(source, [
        row("1234567", "02", "1"),
        row("2345678", "73", "SIM"),
        row("3456789", "05", "N"),
        row("4567890", "05", "SIM", "0"),
        row("5678901", "22", "SIM"),
    ])
    assert import_cnes_csv(source, db, COLUMNS, date(2026, 10, 7), SOURCE) == 2
    provider = CNESFacilityProvider(db)
    ubs = await provider.find_nearby(ServiceType.UBS, -23.55, -46.63, 5)
    upa = await provider.find_nearby(ServiceType.UPA, -23.55, -46.63, 5)
    assert [x.facility.id for x in ubs] == ["cnes-1234567"]
    assert [x.facility.id for x in upa] == ["cnes-2345678"]
    assert not await provider.find_nearby(ServiceType.EMERGENCY_ROOM, -23.55, -46.63, 5)


def test_failed_import_does_not_replace_prior_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "cnes.csv"
    db = tmp_path / "facilities.sqlite"
    write_csv(source, [row("1234567", "02", "SIM")])
    assert import_cnes_csv(source, db, COLUMNS, date(2026, 10, 7), SOURCE) == 1
    before = db.read_bytes()
    write_csv(source, [row("3456789", "05", "N")])
    with pytest.raises(ValueError, match="no eligible"):
        import_cnes_csv(source, db, COLUMNS, date(2026, 10, 7), SOURCE)
    assert db.read_bytes() == before


def test_wrong_schema_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "cnes.csv"
    source.write_text("x;y\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing mapped"):
        import_cnes_csv(
            source, tmp_path / "db.sqlite", COLUMNS, date(2026, 10, 7), SOURCE
        )
