"""Select reproducible train-only IDs; never optimize on the VAE audit validation IDs."""

import argparse
import csv
import json
import random
from pathlib import Path

from phase3.reconstruction_data import file_hash, registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase1-manifest', type=Path, required=True)
    parser.add_argument('--phase2-manifest', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260902)
    args = parser.parse_args()
    train, held_out = registry(args.split_dir)
    with args.phase1_manifest.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with args.phase2_manifest.open(encoding='utf-8-sig', newline='') as handle:
        phase2 = {r['image_id']: r for r in csv.DictReader(handle)}
    eligible, excluded = [], []
    for row in rows:
        name = row['image_id']
        if name not in train:
            continue
        paths = [row.get('image_path'), row.get('deca_mat_path'), row.get('arcface_embedding_path'), phase2.get(name, {}).get('out_npz')]
        if name in held_out or any(not p or not (args.project_root / p).is_file() for p in paths):
            excluded.append(name)
        else:
            eligible.append(name)
    if len(eligible) != len(set(eligible)) or args.count < 1 or len(eligible) < args.count:
        raise ValueError('Invalid eligible pool/count')
    selected = random.Random(args.seed).sample(sorted(eligible), args.count)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    path = args.out_dir / 'train_smoke_ids.txt'
    path.write_text(''.join(f'{name}\n' for name in selected), encoding='utf-8')
    summary = {'selected': selected, 'count': len(selected), 'eligible': len(eligible),
               'excluded_missing_inputs': excluded, 'seed': args.seed, 'validation_fixed_overlap': 0,
               'ids_sha256': file_hash(path), 'phase1_sha256': file_hash(args.phase1_manifest),
               'phase2_sha256': file_hash(args.phase2_manifest), 'scope': 'train-only reconstruction engineering smoke'}
    (args.out_dir / 'selection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
