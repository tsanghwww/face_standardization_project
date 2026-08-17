import argparse
import csv
import json
from pathlib import Path


METRICS_FILE = "gaze_metrics.csv"
SUMMARY_FILE = "gaze_summary.json"


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


def has_gaze_label(row):
    return row.get("gaze_pitch") is not None and row.get("gaze_yaw") is not None


def evaluate(rows, root):
    metric_rows = []
    missing_counts = {
        "source_image": 0,
        "generated_image": 0,
        "gaze_label": 0,
    }

    for row in rows:
        source_exists = exists(row.get("source_image"), root)
        generated_exists = exists(row.get("generated_image"), root)
        gaze_label_exists = has_gaze_label(row)

        if not source_exists:
            missing_counts["source_image"] += 1
        if not generated_exists:
            missing_counts["generated_image"] += 1
        if not gaze_label_exists:
            missing_counts["gaze_label"] += 1

        metric_rows.append(
            {
                "image_id": row.get("image_id"),
                "source_image_exists": source_exists,
                "generated_image_exists": generated_exists,
                "gaze_label_exists": gaze_label_exists,
                "source_gaze_pitch": row.get("gaze_pitch"),
                "source_gaze_yaw": row.get("gaze_yaw"),
                "generated_gaze_pitch": "",
                "generated_gaze_yaw": "",
                "gaze_delta": "",
                "status": "placeholder",
            }
        )

    summary = {
        "metric_family": "gaze_behavior",
        "status": "placeholder",
        "sample_count": len(rows),
        "missing_file_counts": missing_counts,
        "claim_boundary": (
            "This skeleton measures gaze-related behavior only; it does not "
            "claim true gaze disentanglement."
        ),
        "outputs": {
            "metrics_csv": METRICS_FILE,
            "summary_json": SUMMARY_FILE,
        },
        "todo": "Run L2CS or a validated gaze estimator on generated outputs and compare behavior.",
    }
    return metric_rows, summary


def write_metrics(path, rows):
    fieldnames = [
        "image_id",
        "source_image_exists",
        "generated_image_exists",
        "gaze_label_exists",
        "source_gaze_pitch",
        "source_gaze_yaw",
        "generated_gaze_pitch",
        "generated_gaze_yaw",
        "gaze_delta",
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
        description="Prepare placeholder gaze-behavior metrics."
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
