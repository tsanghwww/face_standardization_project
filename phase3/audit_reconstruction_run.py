"""Audit smoke-run files and summarize fixed-noise training diagnostics, not image quality."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(root: Path):
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if summary['frozen_unet_hash_before'] != summary['frozen_unet_hash_after']:
        raise ValueError('Frozen backbone changed')
    if sha256(root/'checkpoint.pt') != summary['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    log = [json.loads(line) for line in (root/'training_log.jsonl').read_text().splitlines() if line.strip()]
    if [row['step'] for row in log] != list(range(1, summary['optimizer_steps']+1)):
        raise ValueError('Missing, duplicate, or unordered optimizer-step logs')
    metrics = {name: summary[name]['mean_epsilon_mse'] for name in
               ('baseline','trained','no_face','no_identity','shuffled_conditions')}
    if not all(np.isfinite(value) for value in metrics.values()) or metrics['baseline'] <= 0:
        raise ValueError('Invalid diagnostic metrics')
    metrics['relative_mse_reduction'] = 1-metrics['trained']/metrics['baseline']
    samples = []
    for path in sorted((root/'samples').glob('*_ddim.png')):
        with Image.open(path) as image:
            array = np.array(image)
            samples.append({'file':path.name, 'size':list(image.size), 'std':float(array.std()), 'sha256':sha256(path)})
    if len(samples) != summary['ddim_samples'] or any(s['size'] != [256,256] or s['std'] <= 1 for s in samples):
        raise ValueError('Missing, wrong-sized, or blank DDIM sample')
    if samples:
        with Image.open(root/'contact_sheet.png') as image:
            image.crop((0,0,image.width,min(image.height,1120))).save(root/'contact_sheet_first4.png')
    report = {'status':'artifact_checks_passed', 'metrics':metrics, 'optimizer_step_logs':len(log),
              'samples':samples, 'nonblank_256_samples':len(samples),
              'gpu_peak_allocated_mb':summary['gpu_peak_allocated_mb'],
              'last_invocation_wall_seconds':summary['wall_seconds'],
              'checkpoint_sha256':summary['checkpoint_sha256'],
              'scope':'nonblank files and train-set epsilon error only; no identity or disentanglement qualification'}
    (root/'run_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    hashes = {path.relative_to(root).as_posix():sha256(path) for path in sorted(root.rglob('*'))
              if path.is_file() and path.name != 'artifact_hashes.json'}
    (root/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'samples'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    audit(parser.parse_args().run_dir)
