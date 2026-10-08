#!/usr/bin/env python3
"""Audit and summarize paired 2x2x2 Health-flow benchmark outputs.

Input: JSONL with one record per (case_id, harness_enabled,
ml_enabled, patient_context_enabled). No patient identifiers or PHI.
Only TEST records belong in the supplied file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("PRIMARY_CARE", "URGENT_CARE", "EMERGENCY")
SETTINGS = tuple(f"E{h}{m}{p}" for h in (0, 1) for m in (0, 1) for p in (0, 1))


def variant(record: dict) -> str:
    vals = (record[k] for k in ("harness_enabled", "ml_enabled", "patient_context_enabled"))
    flags = tuple(vals)
    if any(type(v) is not bool for v in flags):
        raise ValueError("Flags must be JSON booleans")
    return "E" + "".join(str(int(v)) for v in flags)


def load_records(path: Path) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    hashes: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            key = variant(r)
            cid = str(r["case_id"])
            if r["reference_level"] not in CLASSES or r["predicted_final"] not in CLASSES:
                raise ValueError("Unknown care level")
            if r.get("split") != "test":
                raise ValueError("Only test records may be summarized")
            if key in grouped[cid]:
                raise ValueError(f"Duplicate variant {key} for case {cid}")
            if not isinstance(r["dataset_hash"], str) or not r["dataset_hash"]:
                raise ValueError("Missing dataset_hash")
            hashes.add(r["dataset_hash"])
            grouped[cid][key] = r
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"line {lineno}: {exc}") from exc
    if len(hashes) != 1:
        raise ValueError(f"Expected one dataset_hash, found {len(hashes)}")
    if not grouped:
        raise ValueError("No records")
    for cid, variants in grouped.items():
        if set(variants) != set(SETTINGS):
            raise ValueError(f"{cid}: missing or extra variants; found {sorted(variants)}")
        if len({v["reference_level"] for v in variants.values()}) != 1:
            raise ValueError(f"{cid}: conflicting reference levels")
    return grouped


def scores(items: list[dict]) -> dict:
    cm = {t: Counter() for t in CLASSES}
    for r in items:
        cm[r["reference_level"]][r["predicted_final"]] += 1
    f1s = []
    for cls in CLASSES:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in CLASSES if other != cls)
        fn = sum(cm[cls][other] for other in CLASSES if other != cls)
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    emergencies = sum(cm["EMERGENCY"].values())
    correct_emergencies = cm["EMERGENCY"]["EMERGENCY"]
    rank = {c: i for i, c in enumerate(CLASSES)}
    under = sum(rank[r["predicted_final"]] < rank[r["reference_level"]] for r in items)
    return {
        "n": len(items),
        "support": {cls: sum(cm[cls].values()) for cls in CLASSES},
        "f1_macro": statistics.mean(f1s) if all(sum(cm[c].values()) > 0 for c in CLASSES) else None,
        "emergency_recall": correct_emergencies / emergencies if emergencies else None,
        "undertriage_rate": under / len(items),
        "confusion": {cls: [cm[cls][pred] for pred in CLASSES] for cls in CLASSES},
        "latency_ms_p50": statistics.median([r["duration_ms"] for r in items])
        if all(isinstance(r.get("duration_ms"), (int, float)) for r in items) else None,
    }


def summarize(path: Path) -> dict:
    grouped = load_records(path)
    output = {key: scores([cases[key] for cases in grouped.values()]) for key in SETTINGS}
    return {"status": "observed_experimental_results", "n_paired_cases": len(grouped),
            "dataset_hash": next(iter(next(iter(grouped.values())).values()))["dataset_hash"],
            "variants": output,
            "limitations": [
                "No clinical validation implied",
                "Intervals and cluster-adjusted uncertainty not computed",
                "Insufficient class support must be assessed before interpreting metrics",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.jsonl)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
