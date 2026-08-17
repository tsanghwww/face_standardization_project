import argparse
import csv
import json
from pathlib import Path


METRICS_FILE = "identity_metrics.csv"
SUMMARY_FILE = "identity_summary.json"


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
        "source_image": 0,
        "arcface_embedding": 0,
        "generated_image": 0,
    }

    for row in rows:
        source_exists = exists(row.get("source_image"), root)
        embedding_exists = exists(row.get("arcface_embedding"), root)
        generated_exists = exists(row.get("generated_image"), root)

        if not source_exists:
            missing_counts["source_image"] += 1
        if not embedding_exists:
            missing_counts["arcface_embedding"] += 1
        if not generated_exists:
            missing_counts["generated_image"] += 1

        metric_rows.append(
            {
                "image_id": row.get("image_id"),
                "source_image_exists": source_exists,
                "arcface_embedding_exists": embedding_exists,
                "generated_image_exists": generated_exists,
                "arcface_similarity": "",
                "status": "placeholder",
            }
        )

    summary = {
        "metric_family": "identity_preservation",
        "status": "placeholder",
        "sample_count": len(rows),
        "missing_file_counts": missing_counts,
        "outputs": {
            "metrics_csv": METRICS_FILE,
            "summary_json": SUMMARY_FILE,
        },
        "todo": "Compare source and generated outputs with ArcFace embeddings.",
    }
    return metric_rows, summary


def write_metrics(path, rows):
    fieldnames = [
        "image_id",
        "source_image_exists",
        "arcface_embedding_exists",
        "generated_image_exists",
        "arcface_similarity",
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
        description="Prepare placeholder identity-preservation metrics."
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
