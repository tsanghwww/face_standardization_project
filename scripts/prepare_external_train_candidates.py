"""Prepare COFW official-train candidates without modifying canonical registries.

All candidates remain ineligible for training pending identity/crop/DECA quality
checks. Pixel and perceptual duplicate screening is not identity isolation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from collections import Counter

import h5py
import numpy as np
from PIL import Image
from scipy.fft import dctn

from phase3.reconstruction_data import file_hash


def signatures(image):
    im = image.convert('RGB')
    arr = np.asarray(im)
    exact = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
    small = np.asarray(im.convert('L').resize((32, 32)), dtype=np.float32)
    coeff = dctn(small, norm='ortho')[:8, :8].flatten()[1:]
    phash = sum(int(v > np.median(coeff)) << i for i, v in enumerate(coeff))
    return exact, phash


def cofw_image(f, refs, i):
    a = f[refs[0, i]][:]
    return Image.fromarray(a.transpose(2, 1, 0) if a.ndim == 3 else a.T).convert('RGB')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--base-manifest', type=Path, default=Path('results/phase1_parity/phase1_master_manifest.csv'))
    args = p.parse_args()
    root = args.root.resolve()
    if args.out_dir.exists():
        raise ValueError('Output must be new')
    if args.count < 1:
        raise ValueError('Positive count required')
    cofw = root/'datasets/external/COFW_Color/COFW_color'
    train, test = cofw/'COFW_train_color.mat', cofw/'COFW_test_color.mat'
    registry = root/'results/phase30_20260901/splits'
    registry_hashes = {p.name: file_hash(p) for p in registry.glob('*.txt')}
    fixed = root/'results/phase2_eval_fixed_20260824_v2/fixed_test_manifest_v2.csv'
    base_manifest = args.base_manifest if args.base_manifest.is_absolute() else root/args.base_manifest
    validation_ids = {
        line.strip() for line in (registry/'validation_ids.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()
    }
    protected = []
    with h5py.File(test, 'r') as f:
        for i in range(f['IsT'].shape[1]):
            protected.append(signatures(cofw_image(f, f['IsT'], i)))
    with fixed.open(encoding='utf-8-sig', newline='') as f:
        fixed_rows = list(csv.DictReader(f))
    if len(fixed_rows) != 775:
        raise ValueError('Expected complete 775 fixed-test ledger')
    for row in fixed_rows:
        path = Path(row['image_path'])
        if not path.is_absolute():
            path = root/path
        with Image.open(path) as im:
            protected.append(signatures(im))
    with base_manifest.open(encoding='utf-8-sig', newline='') as f:
        base_rows = {str(row['image_id']): row for row in csv.DictReader(f)}
    if validation_ids - set(base_rows):
        raise ValueError('Canonical validation IDs are absent from the Phase1 manifest')
    for image_id in sorted(validation_ids):
        path = Path(base_rows[image_id]['image_path'])
        if not path.is_absolute():
            path = root/path
        with Image.open(path) as im:
            protected.append(signatures(im))
    selected, excluded = [], []
    with h5py.File(train, 'r') as f:
        refs = f['IsTr']
        indices = list(range(refs.shape[1]))
        random.Random(args.seed).shuffle(indices)
        # Screen all official training images; choose first N accepted in frozen random order.
        for i in indices:
            im = cofw_image(f, refs, i)
            exact, phash = signatures(im)
            near = min((phash ^ other).bit_count() for _, other in protected)
            if any(exact == other for other, _ in protected) or near <= 4:
                excluded.append({'index_zero_based': i, 'reason': 'protected_or_selected_pixel/perceptual_duplicate', 'nearest_phash_distance': near})
                continue
            if len(selected) < args.count:
                row = {'image_id': f'exttrain_cofw_{i:04d}', 'source_dataset': 'COFW', 'official_split': 'train',
                       'container': str(train), 'hdf5_key': 'IsTr', 'index_zero_based': i,
                       'source_family': f'COFW/train/{i}', 'pixel_sha256': exact, 'phash63': str(phash),
                       'nearest_protected_phash_distance': near, 'training_eligible': False,
                       'status': 'candidate_pending_identity_crop_geometry_checks'}
                selected.append(row)
                protected.append((exact, phash))
        n_train = refs.shape[1]
    if len(selected) != args.count:
        raise ValueError('Insufficient eligible candidates')
    wlp = root/'datasets/external/300W-LP/extracted/300W_LP'
    counts, families = Counter(), set()
    for group in ('AFW', 'HELEN', 'LFPW', 'IBUG'):
        for path in (wlp/group).glob('*.jpg'):
            counts[group] += 1
            families.add(group+'/'+re.sub(r'_\d+$', '', path.stem))
    args.out_dir.mkdir(parents=True)
    image_dir = args.out_dir/'images'
    image_dir.mkdir()
    with h5py.File(train, 'r') as f:
        refs = f['IsTr']
        for row in selected:
            image = cofw_image(f, refs, int(row['index_zero_based']))
            output = image_dir/f"{row['image_id']}.png"
            image.save(output)
            row['source_image'] = str(output.resolve())
            row['source_image_sha256'] = file_hash(output)
            if row['source_image_sha256'] != row['pixel_sha256']:
                check, _ = signatures(Image.open(output))
                if check != row['pixel_sha256']:
                    raise RuntimeError(f"Materialized image changed pixels: {row['image_id']}")
    (args.out_dir/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected), encoding='utf-8')
    (args.out_dir/'excluded.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded), encoding='utf-8')
    if registry_hashes != {p.name: file_hash(p) for p in registry.glob('*.txt')}:
        raise RuntimeError('Canonical registry changed')
    report = {'status': 'candidates_prepared_not_training_ready', 'n_candidates': len(selected), 'official_train_n': n_train,
              'candidate_ids_test_overlap': 0, 'protected_fixed_rows': len(fixed_rows),
              'protected_validation_rows': len(validation_ids), 'excluded_n': len(excluded),
              'selection_seed': args.seed, 'materialized_images': len(selected),
              'training_steps': 0, 'canonical_registry_hashes_unchanged': registry_hashes,
              'source_hashes': {str(p): file_hash(p) for p in (train, test, fixed, base_manifest)},
              'candidate_manifest_sha256': file_hash(args.out_dir/'candidates.jsonl'),
              'wlp_nonflip_images_by_group': counts, 'wlp_source_family_estimate': len(families),
              'wlp_status': 'not_selected: official original-source train membership and family/identity overlap unresolved',
              'limits': 'pHash whole-image screening cannot establish identity isolation or detect every crop/flip duplicate; ArcFace candidate review and generation-time DECA/Phase2 provenance still required'}
    (args.out_dir/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
