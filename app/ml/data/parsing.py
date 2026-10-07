"""Small parsing helpers shared by raw-dataset loaders (MIMIC-IV-ED, Triagegeist)."""

import gzip
import io
import math
from pathlib import Path
from typing import TextIO


def parse_number(raw: str | None) -> float | None:
    """Finite float from a CSV cell, or None for blanks, text, NaN and infinities."""
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def open_text(path: Path) -> TextIO:
    """Open a `.csv` or `.csv.gz` file as UTF-8 text suitable for `csv.DictReader`."""
    if path.name.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")
