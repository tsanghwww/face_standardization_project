#!/usr/bin/env python3
"""Freeze and audit the Phase3 train/validation/fixed-evaluation partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"ID file not found: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate IDs in {path}")
    return ids


def read_manifest_ids(path: Path, id_column: str, filter_column: str, filter_values: set[str]) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or id_column not in rows[0]:
        raise SystemExit(f"Column {id_column!r} not found in {path}")
    if filter_column:
        if filter_column not in rows[0]:
            raise SystemExit(f"Filter column {filter_column!r} not found in {path}")
        rows = [row for row in rows if row.get(filter_column, "").strip() in filter_values]
    ids = [row[id_column].strip() for row in rows if row[id_column].strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate {id_column} values in {path}")
    return ids


def ids_sha256(ids: list[str]) -> str:
    payload = "".join(f"{image_id}\n" for image_id in sorted(ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_disjoint(groups: dict[str, list[str]]) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            count = len(set(groups[left]) & set(groups[right]))
            overlaps[f"{left}__{right}"] = count
            if count:
                raise SystemExit(f"Split leakage: {left} and {right} overlap by {count} IDs")
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ids", required=True, type=Path)
    parser.add_argument("--val-ids", required=True, type=Path)
    parser.add_argument("--base-test-ids", required=True, type=Path)
    parser.add_argument("--external-manifest", required=True, type=Path)
    parser.add_argument("--external-id-column", default="eval_id")
    parser.add_argument("--external-filter-column", default="")
    parser.add_argument("--external-filter-values", nargs="*", default=[])
    parser.add_argument("--expected-counts", default="", help="Optional train,validation,base,external counts")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    groups = {
        "train": read_ids(args.train_ids),
        "validation": read_ids(args.val_ids),
        "fixed_test_base": read_ids(args.base_test_ids),
        "fixed_test_external": read_manifest_ids(
            args.external_manifest,
            args.external_id_column,
            args.external_filter_column,
            set(args.external_filter_values),
        ),
    }
    if args.external_filter_column and not args.external_filter_values:
        raise SystemExit("--external-filter-column requires --external-filter-values")
    if args.expected_counts:
        expected_values = [int(value) for value in args.expected_counts.split(",")]
        if len(expected_values) != 4:
            raise SystemExit("--expected-counts must be train,validation,base,external")
        expected = dict(zip(groups, expected_values))
        actual = {name: len(ids) for name, ids in groups.items()}
        if actual != expected:
            raise SystemExit(f"Unexpected split counts: expected={expected}, actual={actual}")
    overlaps = require_disjoint(groups)
    fixed_test = groups["fixed_test_base"] + groups["fixed_test_external"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in {**groups, "fixed_test": fixed_test}.items():
        (args.out_dir / f"{name}_ids.txt").write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")

    summary = {
        "protocol": "phase3_split_registry_v1",
        "counts": {name: len(ids) for name, ids in {**groups, "fixed_test": fixed_test}.items()},
        "sha256": {name: ids_sha256(ids) for name, ids in {**groups, "fixed_test": fixed_test}.items()},
        "overlaps": overlaps,
        "roles": {
            "train": "optimization only",
            "validation": "model, loss-weight, and checkpoint selection",
            "fixed_test_base": "final evaluation only",
            "fixed_test_external": "final hard-domain evaluation only; rescue remains audit-only",
        },
    }
    (args.out_dir / "phase3_split_registry.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
