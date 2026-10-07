"""Print a Markdown table of benchmark records (for the article).

Usage: uv run python benchmarks/scripts/summarize_results.py [benchmarks/results]
"""

import json
import sys
from pathlib import Path
from typing import Any

COLUMNS = (
    "timestamp",
    "experiment",
    "dataset_version",
    "model_name",
    "selected",
    "rows",
    "patients",
    "cv_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_emergency_recall",
    "test_under_triage",
    "test_critical_under_triage",
    "train_s",
    "converged",
    "git_commit",
)


def _short_commit(commit: str) -> str:
    # Keep the -dirty marker visible: such runs must not be cited.
    return commit[:12] + (" (dirty)" if commit.endswith("-dirty") else "")


def _converged(record: dict[str, Any]) -> str:
    # Records written before diagnostics existed have no convergence information.
    diagnostics = record.get("training_diagnostics")
    if not diagnostics:
        return "n/a"
    return "yes" if diagnostics["converged"] else "NO"


def summarize(directory: Path) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        cv, test = record["cross_validation_metrics"], record["test_metrics"]
        cells = (
            record["timestamp"][:19],
            record["experiment"],
            record["dataset_version"],
            record["model_name"],
            "yes" if record["selected_for_deployment"] else "",
            record["number_of_rows"],
            record["number_of_patients"] or "n/a",
            f"{cv['macro_f1']:.4f}",
            f"{test['accuracy']:.4f}",
            f"{test['macro_f1']:.4f}",
            f"{test['emergency_recall']:.4f}",
            f"{test['under_triage_rate']:.4f}",
            f"{test['critical_under_triage_rate']:.4f}",
            f"{record['training_time_seconds']:.3f}",
            _converged(record),
            _short_commit(record["git_commit"]),
        )
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/results")
    sys.stdout.write(summarize(target) + "\n")
