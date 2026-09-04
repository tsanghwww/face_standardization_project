"""Full-denominator paired ArcFace and pixel-change audit of latent img2img."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from phase3.audit_vae_roundtrip import load_arcface
from phase3.reconstruction_data import file_hash


def stats(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return {'count': len(values), 'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None,
            'p10': float(np.percentile(values, 10)) if values else None}


def paired_summary(rows, ids, key):
    lookup = {(r['image_id'], r['variant']): r for r in rows}
    pairs = [(lookup.get((i, 'frozen'), {}).get(key), lookup.get((i, 'trained'), {}).get(key)) for i in ids]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return {'n_total': len(ids), 'n_paired': len(pairs),
            'frozen': stats([a for a, b in pairs]), 'trained': stats([b for a, b in pairs]),
            'trained_minus_frozen': stats([b-a for a, b in pairs])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / 'config.json').read_text(encoding='utf-8'))
    generation = json.loads((args.run_dir / 'summary.json').read_text(encoding='utf-8'))
    if generation['samples_sha256'] != file_hash(args.run_dir / 'samples.jsonl'):
        raise ValueError('Sample manifest hash mismatch')
    samples = [json.loads(line) for line in (args.run_dir / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup = {(r['image_id'], r['variant'], r['key']): r for r in samples}
    ids, variants, schedules = config['image_ids'], config['variants'], config['schedules']
    expected = {(i, v, s['key']) for i in ids for v in variants for s in schedules}
    if len(lookup) != len(samples) or set(lookup) != expected:
        raise ValueError('Duplicate, missing or unexpected sample records')
    for i in ids:
        noises = {r['noise_sha256'] for r in samples if r['image_id'] == i}
        if len(noises) != 1:
            raise ValueError('Noise differs between strengths/variants')
        for spec in schedules:
            paired = [lookup[i, v, spec['key']] for v in variants]
            if all(r['status'] == 'generated' for r in paired):
                if len({r['initial_latent_sha256'] for r in paired}) != 1:
                    raise ValueError('Paired initial latent mismatch')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / 'exact_command.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding='utf-8')
    app = load_arcface(0.1)

    def read(path):
        with Image.open(path) as image:
            if image.size != (256, 256):
                raise ValueError(f'Unexpected image dimensions: {path}')
            return np.array(image.convert('RGB'))

    def detect(array):
        try:
            faces = app.get(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            if not faces:
                return None, 0, 'no_face_detected'
            face = max(faces, key=lambda f: float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if embedding.shape != (512,) or not np.isfinite(embedding).all() or norm < 1e-8:
                return None, len(faces), 'invalid_embedding'
            return embedding / norm, len(faces), ''
        except Exception as error:
            return None, None, f'{type(error).__name__}: {error}'

    def cosine(a, b):
        return float(np.dot(a, b)) if a is not None and b is not None else None

    rows, references = [], []
    for image_id in ids:
        images, embeddings, counts, errors = {}, {}, {}, {}
        for ref in ('source', 'vae'):
            try:
                images[ref] = read(args.run_dir / 'references' / f'{image_id}_{ref}.png')
                embeddings[ref], counts[ref], errors[ref] = detect(images[ref])
            except Exception as error:
                images[ref], embeddings[ref], counts[ref], errors[ref] = None, None, None, str(error)
        references.append({'image_id': image_id, 'source_vae_cosine': cosine(embeddings['source'], embeddings['vae']),
                           'source_faces': counts['source'], 'vae_faces': counts['vae'], 'errors': errors})
        for spec in schedules:
            for variant in variants:
                sample = lookup[image_id, variant, spec['key']]
                row = {'image_id': image_id, 'variant': variant, 'key': spec['key'], 'strength': spec['strength'],
                       'source_faces': counts['source'], 'vae_faces': counts['vae'], 'generated_faces': None,
                       'source_cosine': None, 'vae_cosine': None, 'source_mae': None, 'vae_mae': None,
                       'source_rmse': None, 'vae_rmse': None, 'status': 'pending', 'failure_reason': '',
                       'reference_errors': json.dumps(errors), 'source_vae_cosine': references[-1]['source_vae_cosine']}
                try:
                    if sample['status'] != 'generated':
                        raise ValueError(sample['failure_reason'])
                    path = args.run_dir / sample['output']
                    if file_hash(path) != sample['sha256']:
                        raise ValueError('Generated image hash mismatch')
                    generated = read(path)
                    embedding, count, error = detect(generated)
                    row['generated_faces'] = count
                    row['failure_reason'] = error
                    for ref in ('source', 'vae'):
                        row[f'{ref}_cosine'] = cosine(embeddings[ref], embedding)
                        if images[ref] is not None:
                            delta = (generated.astype(np.float32) - images[ref].astype(np.float32)) / 255
                            row[f'{ref}_mae'] = float(np.abs(delta).mean())
                            row[f'{ref}_rmse'] = float(np.sqrt(np.square(delta).mean()))
                    if spec['strength'] == 0 and row['vae_mae'] != 0:
                        raise ValueError('Zero-strength output differs from VAE anchor')
                    row['status'] = 'metrics_available' if all(row[f'{ref}_cosine'] is not None for ref in ('source', 'vae')) else 'detection_failure'
                except Exception as error:
                    row.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
                rows.append(row)
        print(json.dumps({'image_id': image_id, 'audited': len(rows)}), flush=True)
    with (args.out_dir / 'metrics.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_keys = ('source_cosine', 'vae_cosine', 'source_mae', 'vae_mae', 'source_rmse', 'vae_rmse')
    groups, comparisons = [], []
    for spec in schedules:
        subset = [r for r in rows if r['key'] == spec['key']]
        for variant in variants:
            arm = [r for r in subset if r['variant'] == variant]
            groups.append({'variant': variant, **spec, 'n_total': len(ids),
                           'metrics': {k: stats([r[k] for r in arm]) for k in metric_keys},
                           'no_face': sum(r['generated_faces'] == 0 for r in arm),
                           'multiple_faces': sum(r['generated_faces'] is not None and r['generated_faces'] > 1 for r in arm),
                           'detection_failure_rows': sum(r['status'] == 'detection_failure' for r in arm),
                           'failed_rows': sum(r['status'] == 'failed' for r in arm)})
        if 'trained' in variants:
            comparisons.append({**spec, 'metrics': {k: paired_summary(subset, ids, k) for k in metric_keys}})
    summary = {'n_ids': len(ids), 'expected_rows': len(expected), 'audited_rows': len(rows),
               'source_vae_cosine': stats([r['source_vae_cosine'] for r in references]),
               'groups': groups, 'paired_comparisons': comparisons, 'references': references,
               'arcface_model': 'buffalo_l', 'det_thresh': 0.1, 'face_selection': 'largest_bbox; multi-face flags retained',
               'arcface_model_hashes': {str(p): file_hash(p) for p in Path(app.model_dir).glob('*.onnx')},
               'config_sha256': file_hash(args.run_dir / 'config.json'),
               'samples_sha256': file_hash(args.run_dir / 'samples.jsonl'),
               'evaluator_sha256': file_hash(Path(__file__)),
               'scope': '32-train-smoke diagnostic when using original manifest; no held-out tuning or control-success claim',
               'pixel_metrics': 'MAE/RMSE on RGB [0,1], not LPIPS; small changes can simply reflect source copying'}
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'n_ids': len(ids), 'audited_rows': len(rows), 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
