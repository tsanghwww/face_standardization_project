"""ArcFace source/VAE/DDIM audit with full-split denominators and multi-face flags."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL

from phase3.audit_vae_roundtrip import load_arcface
from phase3.reconstruction_data import ReconstructionDataset, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--vae-path', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    dataset = ReconstructionDataset(args.manifest,args.split_dir)
    vae = AutoencoderKL.from_pretrained(args.vae_path,local_files_only=True,torch_dtype=torch.float32).to(args.device).eval().requires_grad_(False)
    app = load_arcface(0.1)

    def detect(rgb):
        faces = app.get(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        if not faces:
            return None, 0
        face = max(faces,key=lambda f: float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        embedding = np.asarray(face.normed_embedding,dtype=np.float32)
        if not np.isfinite(embedding).all():
            return None,len(faces)
        return embedding,len(faces)

    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        row = {'image_id':item['image_id'],'source_faces':None,'vae_faces':None,'ddim_faces':None,
               'source_vae_cosine':None,'source_ddim_cosine':None,'vae_ddim_cosine':None,
               'status':'pending','failure_reason':''}
        try:
            source = ((item['image']+1)*127.5).round().byte().permute(1,2,0).numpy()
            with torch.no_grad():
                z = vae.encode(item['image'][None].to(args.device)).latent_dist.mode()
                decoded = vae.decode(z).sample[0]
                if not torch.isfinite(decoded).all():
                    raise ValueError('Nonfinite VAE anchor')
                anchor = ((decoded.clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
            generated_path = args.run_dir/'samples'/f'{item["image_id"]}_ddim.png'
            with Image.open(generated_path) as image:
                generated = np.array(image.convert('RGB'))
            embeddings = []
            for key,array in [('source',source),('vae',anchor),('ddim',generated)]:
                embedding,count = detect(array)
                embeddings.append(embedding)
                row[f'{key}_faces'] = count
            for a,b,key in [(0,1,'source_vae_cosine'),(0,2,'source_ddim_cosine'),(1,2,'vae_ddim_cosine')]:
                if embeddings[a] is not None and embeddings[b] is not None:
                    row[key] = float(np.dot(embeddings[a],embeddings[b]))
            row['status'] = 'metrics_available' if all(e is not None for e in embeddings) else 'detection_failure'
        except Exception as error:
            row['status'] = 'failed'
            row['failure_reason'] = f'{type(error).__name__}:{error}'
        rows.append(row)
        print(json.dumps(row),flush=True)
    with (args.out_dir/'identity_metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def stats(key):
        values = [r[key] for r in rows if r[key] is not None]
        return {'count':len(values),'mean':float(np.mean(values)) if values else None,
                'median':float(np.median(values)) if values else None}
    summary = {'n_total':len(rows), 'metrics':{key:stats(key) for key in ('source_vae_cosine','source_ddim_cosine','vae_ddim_cosine')},
               'ddim_no_face':sum(r['ddim_faces']==0 for r in rows),
               'ddim_multiple_faces':sum(r['ddim_faces'] is not None and r['ddim_faces']>1 for r in rows),
               'failed_rows':sum(r['status']=='failed' for r in rows),
               'arcface_model':'buffalo_l','det_thresh':0.1,'face_selection':'largest_bbox; multi-face flags retained',
               'checkpoint_sha256':file_hash(args.run_dir/'checkpoint.pt'),
               'scope':'train-sample identity diagnostic; no identity verification threshold or generalization claim'}
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
