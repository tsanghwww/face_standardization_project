import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")

PHASE1_FIELD_CANDIDATES = {
    "source_image": ("source_image", "image_path", "image", "img_path", "input_image"),
    "deca_mat": ("deca_mat", "deca_path", "deca_params", "mat_path"),
    "arcface_embedding": ("arcface_embedding", "arcface_path", "identity_embedding"),
    "quality_score": ("quality_score", "quality", "screen_quality_score"),
    "gaze_pitch": ("gaze_pitch", "pitch"),
    "gaze_yaw": ("gaze_yaw", "yaw"),
    "split": ("split", "dataset_split"),
}

PHASE2_FIELD_CANDIDATES = {
    "phase2_npz": ("phase2_npz", "phase2_path", "npz_path", "output_npz"),
    "phase2_confidence": ("phase2_confidence", "confidence", "alpha"),
    "phase2_reject_score": ("phase2_reject_score", "reject_score"),
}


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_split_file(path):
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_split_assignments(split_dir):
    if split_dir is None:
        return {}

    assignments = {}
    for split in SPLITS:
        for image_id in read_split_file(split_dir / f"{split}.txt"):
            assignments[image_id] = split
    return assignments


def first_value(row, candidates):
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def coerce_number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def path_exists(path_value, root):
    if not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    return path.exists()


def choose_split(phase1_row, split_assignments):
    image_id = phase1_row.get("image_id")
    if image_id in split_assignments:
        return split_assignments[image_id]

    split = first_value(phase1_row, PHASE1_FIELD_CANDIDATES["split"])
    if split in SPLITS:
        return split
    return "train"


def create_sample(phase1_row, phase2_row, split):
    sample = {
        "image_id": phase1_row.get("image_id"),
        "source_image": first_value(phase1_row, PHASE1_FIELD_CANDIDATES["source_image"]),
        "deca_mat": first_value(phase1_row, PHASE1_FIELD_CANDIDATES["deca_mat"]),
        "phase2_npz": None,
        "depth_map": None,
        "normal_map": None,
        "landmark_map": None,
        "arcface_embedding": first_value(phase1_row, PHASE1_FIELD_CANDIDATES["arcface_embedding"]),
        "gaze_pitch": coerce_number(first_value(phase1_row, PHASE1_FIELD_CANDIDATES["gaze_pitch"])),
        "gaze_yaw": coerce_number(first_value(phase1_row, PHASE1_FIELD_CANDIDATES["gaze_yaw"])),
        "quality_score": coerce_number(first_value(phase1_row, PHASE1_FIELD_CANDIDATES["quality_score"])),
        "phase2_confidence": None,
        "phase2_reject_score": None,
        "split": split,
    }

    if phase2_row:
        sample["phase2_npz"] = first_value(phase2_row, PHASE2_FIELD_CANDIDATES["phase2_npz"])
        sample["phase2_confidence"] = coerce_number(
            first_value(phase2_row, PHASE2_FIELD_CANDIDATES["phase2_confidence"])
        )
        sample["phase2_reject_score"] = coerce_number(
            first_value(phase2_row, PHASE2_FIELD_CANDIDATES["phase2_reject_score"])
        )

    # TODO: fill depth_map, normal_map, and landmark_map once those condition
    # artifacts are generated and recorded in a manifest.
    return sample


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_dataset(args):
    root = Path.cwd()
    phase1_rows = read_csv_rows(args.phase1_manifest)
    phase2_rows = read_csv_rows(args.phase2_manifest) if args.phase2_manifest else []
    phase2_by_image_id = {
        row.get("image_id"): row
        for row in phase2_rows
        if row.get("image_id")
    }
    split_assignments = load_split_assignments(args.split_dir)

    rows_by_split = {split: [] for split in SPLITS}
    missing_counts = Counter()
    phase2_matched = 0
    skipped_missing_image_id = 0

    for phase1_row in phase1_rows:
        image_id = phase1_row.get("image_id")
        if not image_id:
            skipped_missing_image_id += 1
            continue

        phase2_row = phase2_by_image_id.get(image_id)
        if phase2_row:
            phase2_matched += 1

        split = choose_split(phase1_row, split_assignments)
        sample = create_sample(phase1_row, phase2_row, split)
        checks = {
            "source_image_exists": path_exists(sample["source_image"], root),
            "deca_mat_exists": path_exists(sample["deca_mat"], root),
            "phase2_npz_exists": path_exists(sample["phase2_npz"], root),
            "arcface_embedding_exists": path_exists(sample["arcface_embedding"], root),
        }
        sample["debug_path_checks"] = checks

        for key, exists in checks.items():
            if not exists:
                missing_counts[key] += 1

        rows_by_split[split].append(sample)

    summary = {
        "dry_run": args.dry_run,
        "phase1_manifest": str(args.phase1_manifest),
        "phase2_manifest": str(args.phase2_manifest) if args.phase2_manifest else None,
        "split_dir": str(args.split_dir) if args.split_dir else None,
        "total_samples": len(phase1_rows),
        "emitted_samples": sum(len(rows_by_split[split]) for split in SPLITS),
        "skipped_missing_image_id": skipped_missing_image_id,
        "phase2_rows": len(phase2_rows),
        "phase2_matched_samples": phase2_matched,
        "split_counts": {split: len(rows_by_split[split]) for split in SPLITS},
        "missing_path_counts": dict(missing_counts),
        "outputs": {
            split: str(args.out_dir / f"{split}.jsonl")
            for split in SPLITS
        },
        "todo": [
            "Populate depth_map, normal_map, and landmark_map when condition maps exist.",
            "Replace placeholder path checks with project-specific manifest columns as schemas stabilize.",
        ],
    }

    for split in SPLITS:
        write_jsonl(args.out_dir / f"{split}.jsonl", rows_by_split[split])
    write_summary(args.out_dir / "dataset_summary.json", summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build downstream condition-dataset JSONL manifests."
    )
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run validation and write placeholder manifests without generating condition artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.phase1_manifest.exists():
        raise SystemExit(f"Phase1 manifest does not exist: {args.phase1_manifest}")
    if args.phase2_manifest and not args.phase2_manifest.exists():
        raise SystemExit(f"Phase2 manifest does not exist: {args.phase2_manifest}")

    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
