"""Offline CNES ingestion tests. Fixture rows are synthetic, not actual facilities."""

from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.schemas.care import ServiceType
from app.tools.cnes_registry import (
    FIELDS,
    OFFICIAL_CNES_COLUMNS,
    CNESFacilityProvider,
    import_cnes_csv,
)

COLUMNS = {key: key for key in FIELDS}
SOURCE = (
    "https://dadosabertos.saude.gov.br/dataset/cnes-cadastro-nacional-de-estabelecimentos-de-saude"
)


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(";".join(FIELDS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def row(cnes: str, kind: str, sus: str, lat: str = "-23.55") -> str:
    return ";".join(
        [
            cnes,
            "Unidade de teste",
            kind,
            "Rua fictícia",
            "1",
            "Centro",
            "São Paulo",
            "SP",
            lat,
            "-46.63",
            sus,
        ]
    )


@pytest.mark.anyio
async def test_import_filters_non_sus_and_invalid_locations(tmp_path: Path) -> None:
    source = tmp_path / "cnes.csv"
    db = tmp_path / "facilities.sqlite"
    write_csv(
        source,
        [
            row("1234567", "02", "1"),
            row("2345678", "73", "SIM"),
            row("3456789", "05", "N"),
            row("4567890", "05", "SIM", "90"),
            row("5678901", "22", "SIM"),
        ],
    )
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
    with pytest.raises(FileExistsError, match="already exists"):
        import_cnes_csv(source, db, COLUMNS, date(2026, 10, 7), SOURCE)
    assert db.read_bytes() == before


def test_wrong_schema_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "cnes.csv"
    source.write_text("x;y\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing mapped"):
        import_cnes_csv(source, tmp_path / "db.sqlite", COLUMNS, date(2026, 10, 7), SOURCE)


@pytest.mark.anyio
async def test_official_zip_header_latin1_and_short_cnes(tmp_path: Path) -> None:
    source = tmp_path / "cnes.zip"
    headers = [*OFFICIAL_CNES_COLUMNS.values(), "CO_MOTIVO_DESAB"]
    fields = {
        "CO_CNES": "27",
        "NO_FANTASIA": "Hospital São José",
        "TP_UNIDADE": "5",
        "NO_LOGRADOURO": "Rua São José",
        "NU_ENDERECO": "42",
        "NO_BAIRRO": "Centro",
        "CO_IBGE": "260290",
        "CO_UF": "26",
        "NU_LATITUDE": "-8,28",
        "NU_LONGITUDE": "-35,03",
        "CO_AMBULATORIAL_SUS": "SIM",
        "CO_MOTIVO_DESAB": "",
    }
    content = ";".join(headers) + "\n" + ";".join(fields[key] for key in headers) + "\n"
    with ZipFile(source, "w") as archive:
        archive.writestr("cnes_estabelecimentos.csv", content.encode("latin-1"))
    database = tmp_path / "cnes.sqlite"
    assert (
        import_cnes_csv(
            source,
            database,
            OFFICIAL_CNES_COLUMNS,
            date(2026, 10, 7),
            SOURCE,
            encoding="latin-1",
        )
        == 1
    )
    matches = await CNESFacilityProvider(database).find_nearby(
        ServiceType.EMERGENCY_ROOM, -8.28, -35.03, 1
    )
    assert matches[0].facility.id == "cnes-0000027"
    assert "Município IBGE 260290" in matches[0].facility.address
    with pytest.raises(FileExistsError):
        import_cnes_csv(
            source, database, OFFICIAL_CNES_COLUMNS, date(2026, 10, 7), SOURCE, encoding="latin-1"
        )
