"""CLI: Kaggle Triagegeist train.csv → canonical processed dataset (`make dataset-triagegeist`).

Never downloads data and never uses Kaggle credentials. Logs aggregate counts and column
names only (no identifiers, no complaints).
"""

import argparse
import json
import logging
import time
from pathlib import Path

from app.ml.data.base import meta_path, write_canonical_csv
from app.ml.data.triagegeist import (
    DEFAULT_TARGET_COLUMN,
    TRIAGE_ACUITY_MAPPING_DESCRIPTION,
    TRIAGEGEIST_SOURCE,
    TriagegeistReport,
    read_train,
    resolve_train_path,
    triagegeist_dataset_info,
)
from app.ml.experiments import TRIAGEGEIST_PROCESSED_PATH
from app.ml.feature_builders import TriagegeistStructuredFeatureBuilder

logger = logging.getLogger(__name__)
DEFAULT_INPUT = Path("data/raw/triagegeist")


def prepare(
    input_path: Path,
    output_path: Path,
    source_version: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> TriagegeistReport:
    train_path = resolve_train_path(input_path)
    report = TriagegeistReport()
    info = triagegeist_dataset_info(source_version)
    started = time.perf_counter()
    write_canonical_csv(output_path, info, read_train(train_path, report, target_column))
    builder = TriagegeistStructuredFeatureBuilder
    # The sidecar is re-written with the aggregate report, known only after streaming.
    meta_file = meta_path(output_path)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta |= {
        "dataset_name": info.name,
        "dataset_version": info.version,
        "source": TRIAGEGEIST_SOURCE,
        "source_file": train_path.name,
        "number_of_rows": report.rows_kept,
        "number_of_patients": report.number_of_patients,
        "class_distribution": dict(report.class_distribution),
        "discarded_rows": {
            "total": report.rows_read - report.rows_kept,
            "by_reason": dict(report.dropped),
        },
        "mapping_used": {
            "target_column": target_column,
            "table": {
                "1": "EMERGENCY",
                "2": "EMERGENCY",
                "3": "URGENT_CARE",
                "4": "PRIMARY_CARE",
                "5": "PRIMARY_CARE",
            },
            "description": TRIAGE_ACUITY_MAPPING_DESCRIPTION,
            "invalid_values": "missing or outside 1-5 → row discarded (no class invented)",
        },
        "feature_set": builder.feature_set,
        "features": list(builder.feature_names),
        "preparation": {
            **report.as_dict(),
            "duration_seconds": round(time.perf_counter() - started, 2),
        },
    }
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle Triagegeist for training")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="train.csv or the directory with it"
    )
    parser.add_argument("--output", type=Path, default=TRIAGEGEIST_PROCESSED_PATH)
    parser.add_argument(
        "--source-version", default="unspecified", help="Kaggle dataset version/date"
    )
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started = time.perf_counter()
    report = prepare(args.input, args.output, args.source_version, args.target_column)
    stats = report.as_dict()
    logger.info("column mapping=%s", json.dumps(stats["column_mapping"], ensure_ascii=False))
    logger.info(
        "rows read=%d kept=%d dropped=%s", report.rows_read, report.rows_kept, stats["rows_dropped"]
    )
    logger.info(
        "patients=%s acuity=%s classes=%s",
        report.number_of_patients if report.number_of_patients is not None else "n/a (no id)",
        stats["original_acuity_distribution"],
        stats["class_distribution"],
    )
    logger.info("missing after cleaning=%s", stats["missing_after_cleaning"])
    logger.info("implausible→missing=%s", stats["implausible_set_to_missing"])
    if report.number_of_patients is None:
        logger.warning(
            "no patient identifier column: split will be row-level, "
            "WITHOUT protection against patient leakage"
        )
    logger.info("label mapping: %s", TRIAGE_ACUITY_MAPPING_DESCRIPTION)
    logger.info("written %s in %.1fs", args.output, time.perf_counter() - started)


if __name__ == "__main__":
    main()
