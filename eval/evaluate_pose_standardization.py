import argparse
import csv
import json
from pathlib import Path


METRICS_FILE = "pose_metrics.csv"
SUMMARY_FILE = "pose_summary.json"


def load_manifest(path):
    if not path.exists():
        return [], False
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows, True


def exists(path_value, root):
    if not path_value:
        return False
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    return path.exists()


def evaluate(rows, root):
    metric_rows = []
    missing_counts = {
        "deca_mat": 0,
        "phase2_npz": 0,
        "generated_image": 0,
    }

    for row in rows:
        deca_exists = exists(row.get("deca_mat"), root)
        phase2_exists = exists(row.get("phase2_npz"), root)
        generated_exists = exists(row.get("generated_image"), root)

        if not deca_exists:
            missing_counts["deca_mat"] += 1
        if not phase2_exists:
            missing_counts["phase2_npz"] += 1
        if not generated_exists:
            missing_counts["generated_image"] += 1

        metric_rows.append(
            {
                "image_id": row.get("image_id"),
                "deca_mat_exists": deca_exists,
                "phase2_npz_exists": phase2_exists,
                "generated_image_exists": generated_exists,
                "source_pose_error": "",
                "generated_pose_error": "",
                "pose_error_delta": "",
                "status": "placeholder",
            }
        )

    summary = {
        "metric_family": "pose_standardization",
        "status": "placeholder",
        "sample_count": len(rows),
        "missing_file_counts": missing_counts,
        "outputs": {
            "metrics_csv": METRICS_FILE,
            "summary_json": SUMMARY_FILE,
        },
        "todo": "Estimate whether generated outputs move closer to canonical head pose.",
    }
    return metric_rows, summary


def write_metrics(path, rows):
    fieldnames = [
        "image_id",
        "deca_mat_exists",
        "phase2_npz_exists",
        "generated_image_exists",
        "source_pose_error",
        "generated_pose_error",
        "pose_error_delta",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, summary):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare placeholder pose-standardization metrics."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest_exists = load_manifest(args.manifest)
    metric_rows, summary = evaluate(rows, Path.cwd())
    summary["manifest"] = str(args.manifest)
    summary["manifest_exists"] = manifest_exists

    write_metrics(args.out_dir / METRICS_FILE, metric_rows)
    write_summary(args.out_dir / SUMMARY_FILE, summary)


if __name__ == "__main__":
    main()
