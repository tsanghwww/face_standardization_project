"""Held-out geometry-audit inputs kept separate from training fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from phase3.reconstruction_data import file_hash, read_ids


CONDITION_FIELDS = {
    "source": ("source_normal_map", "source_depth_map", "source_landmark_map", "source_face_mask"),
    "target": ("target_normal_map", "target_depth_map", "target_landmark_map", "target_face_mask"),
}
REQUIRED_FIELDS = ("source_image", "deca_mat", "phase2_npz", "arcface_embedding")


def load_condition(row: dict, prefix: str, size: int = 256) -> torch.Tensor:
    if prefix not in CONDITION_FIELDS:
        raise ValueError(f"Unknown condition prefix: {prefix}")
    normal_field, depth_field, landmark_field, mask_field = CONDITION_FIELDS[prefix]
    shape = (size, size)
    with Image.open(row[normal_field]) as image:
        normal = np.asarray(image.convert("RGB").resize(shape, Image.Resampling.BILINEAR), dtype=np.float32) / 255
    with Image.open(row[depth_field]) as image:
        depth = np.asarray(image, dtype=np.float32)
        if depth.min() < 0 or depth.max() > 65535:
            raise ValueError(f"Invalid uint16 depth: {row['image_id']}:{prefix}")
        depth = np.asarray(Image.fromarray(depth / 65535).resize(shape, Image.Resampling.BILINEAR))
    gray = []
    for field in (landmark_field, mask_field):
        with Image.open(row[field]) as image:
            method = Image.Resampling.NEAREST if field == mask_field else Image.Resampling.BILINEAR
            gray.append(np.asarray(image.convert("L").resize(shape, method), dtype=np.float32) / 255)
    value = np.concatenate([normal, depth[..., None], *(item[..., None] for item in gray)], axis=-1)
    if value.shape != (size, size, 6) or not np.isfinite(value).all():
        raise ValueError(f"Invalid condition tensor: {row['image_id']}:{prefix}")
    return torch.from_numpy(value.transpose(2, 0, 1).copy())


class GeometryAuditDataset:
    """Strict train/validation evaluator that always excludes the fixed test set."""

    def __init__(self, manifest: Path, split_dir: Path, ids_file: Path, split: str = "validation", size: int = 256):
        if split not in ("train", "validation"):
            raise ValueError("Geometry audit supports only train or validation; fixed test is sealed")
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {str(row["image_id"]): row for row in rows}
        if not rows or len(by_id) != len(rows):
            raise ValueError("Empty or duplicate evaluation manifest IDs")
        ordered_ids = [line.strip() for line in ids_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("Empty or duplicate evaluation IDs")
        train = read_ids(split_dir / "train_ids.txt")
        validation = read_ids(split_dir / "validation_ids.txt")
        fixed = read_ids(split_dir / "fixed_test_ids.txt")
        expected = train if split == "train" else validation
        if set(ordered_ids) - expected:
            raise ValueError(f"Evaluation IDs are not all in {split}")
        if set(ordered_ids) & fixed:
            raise ValueError("Fixed-test leakage into geometry audit")
        if set(ordered_ids) - set(by_id):
            raise ValueError("Evaluation IDs absent from manifest")

        aliases = {"train"} if split == "train" else {"val", "validation"}
        self.rows = [by_id[image_id] for image_id in ordered_ids]
        self.size = size
        self.input_hashes = []
        fields = (*REQUIRED_FIELDS, *CONDITION_FIELDS["source"], *CONDITION_FIELDS["target"])
        for row in self.rows:
            image_id = str(row["image_id"])
            if row.get("split") not in aliases:
                raise ValueError(f"Manifest split mismatch: {image_id}:{row.get('split')}")
            if row.get("rescue_source") not in (False, None):
                raise ValueError(f"Rescue is forbidden: {image_id}")
            if row.get("condition_cache_status") not in ("ready", "geometry_ready_gaze_pending"):
                raise ValueError(f"Geometry unavailable: {image_id}")
            for field in fields:
                value = row.get(field)
                if not value or not Path(value).is_file():
                    raise ValueError(f"Missing {field}: {image_id}")
                self.input_hashes.append({"image_id": image_id, "field": field, "sha256": file_hash(Path(value))})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(row["source_image"]) as image:
            rgb = np.asarray(image.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        embedding = np.load(row["arcface_embedding"], allow_pickle=False).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.shape != (512,) or not np.isfinite(embedding).all() or norm < 1e-8:
            raise ValueError(f"Invalid ArcFace embedding: {row['image_id']}")
        return {
            "image_id": str(row["image_id"]),
            "image": torch.from_numpy(rgb.transpose(2, 0, 1).copy()) * 2 - 1,
            "source_condition": load_condition(row, "source", self.size),
            "target_condition": load_condition(row, "target", self.size),
            "identity": torch.from_numpy(embedding / norm),
            "deca_mat": row["deca_mat"],
            "phase2_npz": row["phase2_npz"],
        }


def evaluation_fingerprint(dataset: GeometryAuditDataset, manifest: Path, ids_file: Path, split_dir: Path, split: str) -> dict:
    return {
        "scope": "evaluation_only",
        "split": split,
        "manifest_sha256": file_hash(manifest),
        "ids_sha256": file_hash(ids_file),
        "registry_hashes": {
            name: file_hash(split_dir / name)
            for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")
        },
        "inputs": dataset.input_hashes,
    }


def verify_training_isolation(training_fingerprint: dict, evaluation_ids: set[str], split_dir: Path) -> set[str]:
    current_hashes = {
        name: file_hash(split_dir / name)
        for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")
    }
    if training_fingerprint.get("split_hashes") != current_hashes:
        raise ValueError("Checkpoint split registry differs from the evaluation registry")
    training_ids = {str(row["image_id"]) for row in training_fingerprint.get("inputs", ())}
    train_registry = read_ids(split_dir / "train_ids.txt")
    fixed_registry = read_ids(split_dir / "fixed_test_ids.txt")
    if not training_ids or training_ids - train_registry:
        raise ValueError("Checkpoint inputs are not a nonempty subset of canonical train")
    if training_ids & evaluation_ids:
        raise ValueError("Checkpoint training IDs overlap evaluation IDs")
    if training_ids & fixed_registry:
        raise ValueError("Checkpoint training IDs overlap fixed test")
    return training_ids


def verify_target_provenance(path: Path, evaluation: dict) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "verified":
        raise ValueError(f"Target provenance is not strictly verified: {report.get('status')}")
    if report.get("evaluation_manifest_sha256") != evaluation.get("manifest_sha256"):
        raise ValueError("Target provenance evaluation manifest mismatch")
    if report.get("ids_sha256") != evaluation.get("ids_sha256"):
        raise ValueError("Target provenance ID selection mismatch")
    if report.get("condition_lineage_mode") != "generation_time":
        raise ValueError("Target conditions lack generation-time lineage")
    return report


def intervention(items: list[dict], index: int, arm: str) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Return condition/identity inputs while keeping geometry-only arms identity-fixed."""
    item = items[index]
    shuffled = items[(index + 1) % len(items)]
    if arm == "source_geometry":
        return item["source_condition"], item["identity"], item["image_id"], item["image_id"]
    if arm == "target_geometry":
        return item["target_condition"], item["identity"], item["image_id"], item["image_id"]
    if arm == "zero_geometry":
        return torch.zeros_like(item["source_condition"]), item["identity"], "zero", item["image_id"]
    if arm == "shuffled_geometry":
        return shuffled["target_condition"], item["identity"], shuffled["image_id"], item["image_id"]
    if arm == "target_geometry_shuffled_identity":
        return item["target_condition"], shuffled["identity"], item["image_id"], shuffled["image_id"]
    raise ValueError(f"Unknown intervention arm: {arm}")
