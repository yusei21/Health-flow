"""CLI for importing a reviewed CNES CSV with an explicit JSON column map."""

import argparse
import json
from datetime import date
from pathlib import Path

from app.tools.cnes_registry import import_cnes_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("data/cnes.sqlite"))
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--reference-date", type=date.fromisoformat, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--delimiter", default=";", choices=[";", ",", "\t"])
    args = parser.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
    ):
        parser.error("--mapping must contain a JSON object mapping logical fields to CSV headers")
    count = import_cnes_csv(
        args.csv, args.database, mapping, args.reference_date, args.source_url,
        args.delimiter,
    )
    print(f"CNES: {count} eligible, geocoded SUS rows imported to {args.database}")


if __name__ == "__main__":
    main()
