"""FAKE triage files shaped like MIMIC-IV-ED `triage` (same column names).

Every value is randomly generated. This is NOT MIMIC data and contains no real patient.
Includes the messy cases the loader must handle: blanks, text pain, invalid acuity,
implausible vitals and repeated visits of the same subject.
"""

import csv
import gzip
import io
import random
from pathlib import Path

TRIAGE_COLUMNS = (
    "subject_id",
    "stay_id",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
    "chiefcomplaint",
)


def fake_triage_rows(n_rows: int, n_subjects: int, seed: int = 0) -> list[dict[str, str]]:
    rng = random.Random(seed)  # noqa: S311 - fake fixture data
    rows = []
    for stay in range(n_rows):
        acuity = rng.choice([1, 2, 2, 3, 3, 3, 4, 4, 5])
        sick = acuity <= 2
        rows.append(
            {
                "subject_id": str(10_000_000 + rng.randrange(n_subjects)),
                "stay_id": str(30_000_000 + stay),
                "temperature": _maybe(rng, f"{rng.gauss(100.5 if sick else 98.4, 1.0):.1f}"),
                "heartrate": _maybe(rng, f"{rng.gauss(115 if sick else 82, 12):.0f}"),
                "resprate": _maybe(rng, f"{rng.gauss(24 if sick else 16, 3):.0f}"),
                "o2sat": _maybe(rng, f"{min(100, rng.gauss(91 if sick else 98, 2)):.0f}"),
                "sbp": _maybe(rng, f"{rng.gauss(150 if sick else 125, 15):.0f}"),
                "dbp": _maybe(rng, f"{rng.gauss(90 if sick else 78, 10):.0f}"),
                "pain": rng.choice([str(rng.randint(0, 10))] * 8 + ["", "unable"]),
                "acuity": f"{acuity}.0" if rng.random() > 0.02 else rng.choice(["", "7", "2.5"]),
                "chiefcomplaint": "FAKE complaint",
            }
        )
    return rows


def _maybe(rng: random.Random, value: str) -> str:
    return "" if rng.random() < 0.05 else value


def write_fake_triage(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else TRIAGE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(data))
    else:
        path.write_bytes(data)
    return path


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1])
    write_fake_triage(target, fake_triage_rows(int(sys.argv[2]), int(sys.argv[3])))
