"""Strict source-only reconstruction inputs for a train-split engineering smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> set[str]:
    values = [s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f'Empty or duplicate IDs: {path}')
    return set(values)


def registry(root: Path) -> tuple[set[str], set[str]]:
    train = read_ids(root / 'train_ids.txt')
    held_out = read_ids(root / 'validation_ids.txt') | read_ids(root / 'fixed_test_ids.txt')
    if train & held_out:
        raise ValueError('Train registry overlaps validation/fixed test')
    return train, held_out


SOURCE_FIELDS = ('source_image', 'source_normal_map', 'source_depth_map',
                 'source_landmark_map', 'source_face_mask', 'arcface_embedding')


class ReconstructionDataset(Dataset):
    def __init__(self, manifest: Path, split_dir: Path, size: int = 256):
        self.rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.size = size
        train, held_out = registry(split_dir)
        ids = [row['image_id'] for row in self.rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Empty or duplicate reconstruction IDs')
        self.hashes = []
        for row in self.rows:
            name = row['image_id']
            if name not in train or name in held_out or row['split'] != 'train':
                raise ValueError(f'Non-training ID: {name}')
            if row.get('rescue_source') not in (False, None):
                raise ValueError(f'Rescue not allowed: {name}')
            if row.get('condition_cache_status') not in ('ready', 'geometry_ready_gaze_pending'):
                raise ValueError(f'Geometry unavailable: {name}')
            for field in SOURCE_FIELDS:
                value = row.get(field)
                if not value or not Path(value).is_file():
                    raise ValueError(f'Missing {field}: {name}')
                self.hashes.append({'image_id': name, 'field': field, 'sha256': file_hash(Path(value))})
        source_hashes = [r['sha256'] for r in self.hashes if r['field'] == 'source_image']
        if len(source_hashes) != len(set(source_hashes)):
            raise ValueError('Duplicate source image contents')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        size = (self.size, self.size)
        with Image.open(row['source_image']) as image:
            rgb = np.array(image.convert('RGB').resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        with Image.open(row['source_normal_map']) as image:
            normal = np.array(image.convert('RGB').resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255
        with Image.open(row['source_depth_map']) as image:
            depth = np.array(image, dtype=np.float32)
            if depth.max() > 65535 or depth.min() < 0:
                raise ValueError('Invalid uint16 depth')
            depth = np.array(Image.fromarray(depth / 65535).resize(size, Image.Resampling.BILINEAR))
        grayscale = []
        for field in ('source_landmark_map', 'source_face_mask'):
            with Image.open(row[field]) as image:
                method = Image.Resampling.NEAREST if field.endswith('mask') else Image.Resampling.BILINEAR
                grayscale.append(np.array(image.convert('L').resize(size, method), dtype=np.float32) / 255)
        condition = np.concatenate([normal, depth[..., None], *(a[..., None] for a in grayscale)], axis=-1)
        embedding = np.load(row['arcface_embedding'], allow_pickle=False).astype(np.float32).reshape(-1)
        if embedding.shape != (512,) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) < 1e-8:
            raise ValueError(f'Invalid ArcFace embedding: {row["image_id"]}')
        return {
            'image_id': row['image_id'],
            'image': torch.from_numpy(rgb.transpose(2, 0, 1).copy()) * 2 - 1,
            'condition': torch.from_numpy(condition.transpose(2, 0, 1).copy()),
            'identity': torch.from_numpy(embedding / np.linalg.norm(embedding)),
        }
