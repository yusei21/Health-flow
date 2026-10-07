"""CLI: generate the synthetic baseline dataset (`make dataset-synthetic`)."""

import argparse
import logging
from pathlib import Path

from app.ml.data.synthetic import (
    DATA_DISCLAIMER,
    DEFAULT_DATASET_PATH,
    generate_examples,
    save_dataset,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate {DATA_DISCLAIMER}")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    save_dataset(generate_examples(args.rows, args.seed), args.output)
    logger.info("dataset written: %s (%s rows) — %s", args.output, args.rows, DATA_DISCLAIMER)


if __name__ == "__main__":
    main()
