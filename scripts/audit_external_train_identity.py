"""ArcFace identity-isolation audit for a quarantined external training pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from phase3.audit_vae_roundtrip import load_arcface
from phase3.reconstruction_data import file_hash, read_ids


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def detect(app, path: Path) -> tuple[np.ndarray | None, str, float | None]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    faces = app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not faces:
        return None, "no_face_detected", None
    face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.shape != (512,) or not np.isfinite(embedding).all() or norm < 1e-8:
        return None, "invalid_embedding", None
    score = getattr(face, "det_score", None)
    return embedding / norm, "success", float(score) if score is not None and np.isfinite(score) else None


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--base-manifest", type=Path, default=Path("results/phase1_parity/phase1_master_manifest.csv"))
    parser.add_argument("--fixed-manifest", type=Path, default=Path("results/phase2_eval_fixed_20260824_v2/fixed_test_manifest_v2.csv"))
    parser.add_argument("--split-dir", type=Path, default=Path("results/phase30_20260901/splits"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--review-cosine", type=float, default=0.30)
    parser.add_argument("--reject-cosine", type=float, default=0.40)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("Identity-audit output directory must be new")
    if not -1 <= args.review_cosine < args.reject_cosine <= 1:
        raise ValueError("Require -1 <= review-cosine < reject-cosine <= 1")
    root = args.root.resolve()
    candidates_path = resolve(root, str(args.candidates))
    base_path = resolve(root, str(args.base_manifest))
    fixed_path = resolve(root, str(args.fixed_manifest))
    split_dir = resolve(root, str(args.split_dir))

    candidates = jsonl(candidates_path)
    candidate_ids = [str(row["image_id"]) for row in candidates]
    if not candidates or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate manifest is empty or duplicated")
    fixed_ids = read_ids(split_dir / "fixed_test_ids.txt")
    validation_ids = read_ids(split_dir / "validation_ids.txt")
    if set(candidate_ids) & (fixed_ids | validation_ids):
        raise ValueError("External candidate ID overlaps a protected canonical split")

    with base_path.open(encoding="utf-8-sig", newline="") as handle:
        base = {str(row["image_id"]): row for row in csv.DictReader(handle)}
    with fixed_path.open(encoding="utf-8-sig", newline="") as handle:
        fixed_rows = list(csv.DictReader(handle))
    if len(fixed_rows) != 775:
        raise ValueError("Expected the complete 775-row fixed-test ledger")

    protected_vectors, protected_labels = [], []
    protected_failures = []
    base_protected = validation_ids | {str(row["image_id"]) for row in fixed_rows if row["source_dataset"] == "stylegan2_base"}
    for image_id in sorted(base_protected):
        row = base.get(image_id)
        embedding_path = resolve(root, row.get("arcface_embedding_path", "")) if row else Path()
        if not row or row.get("arcface_status") != "success" or not embedding_path.is_file():
            protected_failures.append({"protected_id": image_id, "source": "base", "reason": "missing_valid_embedding"})
            continue
        value = np.load(embedding_path, allow_pickle=False).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if value.shape != (512,) or not np.isfinite(value).all() or norm < 1e-8:
            protected_failures.append({"protected_id": image_id, "source": "base", "reason": "invalid_embedding"})
            continue
        protected_vectors.append(value / norm)
        protected_labels.append(f"base:{image_id}")

    app = load_arcface(args.det_thresh)
    for row in fixed_rows:
        if row["source_dataset"] == "stylegan2_base":
            continue
        label = str(row["eval_id"])
        path = resolve(root, row["image_path"])
        try:
            embedding, status, _ = detect(app, path)
        except Exception as error:
            embedding, status = None, f"{type(error).__name__}: {error}"
        if embedding is None:
            protected_failures.append({"protected_id": label, "source": "fixed_external", "reason": status})
        else:
            protected_vectors.append(embedding)
            protected_labels.append(f"fixed_external:{label}")
    if not protected_vectors:
        raise ValueError("No protected identity embeddings are available")
    protected_matrix = np.stack(protected_vectors)

    args.out_dir.mkdir(parents=True)
    embedding_dir = args.out_dir / "embeddings"
    embedding_dir.mkdir()
    detected = {}
    records = []
    for row in candidates:
        image_id = str(row["image_id"])
        path = resolve(root, row["source_image"])
        if file_hash(path) != row["source_image_sha256"]:
            raise ValueError(f"Candidate image hash mismatch: {image_id}")
        try:
            embedding, status, score = detect(app, path)
        except Exception as error:
            embedding, status, score = None, f"{type(error).__name__}: {error}", None
        record = {
            **row,
            "arcface_status": status,
            "arcface_detector_score": score,
            "embedding_path": "",
            "max_protected_cosine": None,
            "nearest_protected_id": "",
            "max_candidate_cosine": None,
            "nearest_candidate_id": "",
            "identity_decision": "unresolved_no_embedding",
            "training_eligible": False,
        }
        if embedding is not None:
            output = embedding_dir / f"{image_id}.npy"
            np.save(output, embedding, allow_pickle=False)
            record["embedding_path"] = str(output.resolve())
            similarities = protected_matrix @ embedding
            index = int(np.argmax(similarities))
            record["max_protected_cosine"] = float(similarities[index])
            record["nearest_protected_id"] = protected_labels[index]
            detected[image_id] = embedding
        records.append(record)
        print(json.dumps({"image_id": image_id, "arcface_status": status}), flush=True)

    for record in records:
        image_id = record["image_id"]
        embedding = detected.get(image_id)
        if embedding is None:
            continue
        others = [(other_id, value) for other_id, value in detected.items() if other_id != image_id]
        if others:
            values = np.asarray([float(np.dot(embedding, value)) for _, value in others])
            index = int(np.argmax(values))
            record["max_candidate_cosine"] = float(values[index])
            record["nearest_candidate_id"] = others[index][0]
        worst = max(
            value for value in (record["max_protected_cosine"], record["max_candidate_cosine"]) if value is not None
        )
        if worst >= args.reject_cosine:
            record["identity_decision"] = "reject_potential_identity_overlap"
        elif worst >= args.review_cosine:
            record["identity_decision"] = "manual_review_similarity_band"
        else:
            record["identity_decision"] = "machine_pass_conditional_on_protected_coverage"

    manifest_path = args.out_dir / "identity_audit.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    failures_path = args.out_dir / "protected_embedding_failures.jsonl"
    failures_path.write_text("".join(json.dumps(row) + "\n" for row in protected_failures), encoding="utf-8")
    counts = {}
    for row in records:
        counts[row["identity_decision"]] = counts.get(row["identity_decision"], 0) + 1
    report = {
        "status": "identity_audited_not_training_ready",
        "n_candidates": len(records),
        "candidate_arcface_success": len(detected),
        "identity_decisions": counts,
        "protected_expected": len(base_protected) + sum(row["source_dataset"] != "stylegan2_base" for row in fixed_rows),
        "protected_embedding_success": len(protected_vectors),
        "protected_embedding_failures": len(protected_failures),
        "review_cosine": args.review_cosine,
        "reject_cosine": args.reject_cosine,
        "threshold_policy": "frozen diagnostic triage; not an identity non-overlap guarantee",
        "training_eligible_count": 0,
        "next_gate": "FAN-only DECA extraction, quality audit, and generation-time Phase2 target provenance",
        "candidate_manifest_sha256": file_hash(candidates_path),
        "identity_audit_sha256": file_hash(manifest_path),
        "protected_failures_sha256": file_hash(failures_path),
        "split_hashes": {name: file_hash(split_dir / name) for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
