"""CLI: MIMIC-IV-ED triage → canonical processed dataset (`make dataset-mimic`).

Never downloads data. Logs aggregate counts only (no identifiers, no complaints).
"""

import argparse
import json
import logging
import time
from pathlib import Path

from app.ml.data.base import write_canonical_csv
from app.ml.data.mimic_ed import (
    ESI_MAPPING_DESCRIPTION,
    PreparationReport,
    mimic_dataset_info,
    read_triage,
    resolve_triage_path,
)
from app.ml.experiments import MIMIC_PROCESSED_PATH

logger = logging.getLogger(__name__)
DEFAULT_INPUT = Path("data/raw/mimic-iv-ed")


def prepare(input_path: Path, output_path: Path, source_version: str) -> PreparationReport:
    triage_path = resolve_triage_path(input_path)
    report = PreparationReport()
    started = time.perf_counter()
    write_canonical_csv(
        output_path,
        mimic_dataset_info(source_version),
        read_triage(triage_path, report),
        extra_meta={"source_file": triage_path.name, "preparation": None},
    )
    summary = {**report.as_dict(), "duration_seconds": round(time.perf_counter() - started, 2)}
    # Re-write the sidecar with the final aggregate report (known only after streaming).
    meta_file = output_path.with_suffix(".meta.json")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["preparation"] = summary
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIMIC-IV-ED triage for training")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="triage.csv.gz, triage.csv or the directory containing it",
    )
    parser.add_argument("--output", type=Path, default=MIMIC_PROCESSED_PATH)
    parser.add_argument(
        "--source-version", default="unspecified", help="MIMIC-IV-ED release, e.g. 2.2"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.perf_counter()
    report = prepare(args.input, args.output, args.source_version)
    stats = report.as_dict()
    logger.info(
        "rows read=%d kept=%d dropped=%s", report.rows_read, report.rows_kept, stats["rows_dropped"]
    )
    logger.info("patients=%d classes=%s", stats["number_of_patients"], stats["class_distribution"])
    logger.info("missing after cleaning=%s", stats["missing_after_cleaning"])
    logger.info(
        "implausible→missing=%s pain invalid→missing=%d",
        stats["implausible_set_to_missing"],
        report.pain_invalid,
    )
    if report.absent_columns:
        logger.warning("absent optional columns: %s", report.absent_columns)
    logger.info("label mapping: %s", ESI_MAPPING_DESCRIPTION)
    logger.info("written %s in %.1fs", args.output, time.perf_counter() - started)


if __name__ == "__main__":
    main()
