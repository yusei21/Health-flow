"""CLI for importing a reviewed CNES CSV with an explicit JSON column map."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from app.tools.cnes_registry import OFFICIAL_CNES_COLUMNS, import_cnes_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("data/cnes.sqlite"))
    mapping_group = parser.add_mutually_exclusive_group(required=True)
    mapping_group.add_argument("--mapping", type=Path)
    mapping_group.add_argument("--official-establishments", action="store_true")
    parser.add_argument("--reference-date", type=date.fromisoformat, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--delimiter", default=";", choices=[";", ",", "\t"])
    parser.add_argument("--encoding", default=None)
    args = parser.parse_args()
    mapping = (
        OFFICIAL_CNES_COLUMNS
        if args.official_establishments
        else json.loads(args.mapping.read_text(encoding="utf-8"))
    )
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
    ):
        parser.error("--mapping must contain a JSON object mapping logical fields to CSV headers")
    count = import_cnes_csv(
        args.csv,
        args.database,
        mapping,
        args.reference_date,
        args.source_url,
        args.delimiter,
        args.encoding or ("latin-1" if args.official_establishments else "utf-8-sig"),
    )
    sys.stdout.write(f"CNES: {count} eligible, geocoded SUS rows imported to {args.database}\n")


if __name__ == "__main__":
    main()
