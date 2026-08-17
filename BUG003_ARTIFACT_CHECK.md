# BUG-003 ARTIFACT CHECK

Check date: 2026-08-07

Machine checked: `win-lenovo`

Project path: `D:\face_standardization_project`

## Question

The recovered Phase2 inference outputs almost entirely reject samples. This matches the historical BUG-003 symptom: DECA `kpt2d` values were stored in normalized orthographic coordinates, but Phase2 quality scoring treated them as 224px image coordinates.

The goal of this check was to determine whether the nearly-all-reject outputs are stale bug-era artifacts, and whether a corrected full output version already exists on 5060.

## Code State

`phase2/features.py` on 5060 contains the fix:

```python
xy = xy * (image_size / 2.0) + (image_size / 2.0)
```

The project logs also record:

- BUG-003 diagnosed on 2026-07-08 21:53.
- Root cause: `kpt2d` normalized `[-1, 1]` vs Phase2 quality code expecting pixel coordinates.
- Resolution: denormalization added to `phase2/features.py`.
- `docs/STATUS.md` records a 200-sample spot check after the fix:
  - `landmark_score` mean 0.943
  - `landmark_out_ratio` 0.000
  - `quality_score` mean 0.683

## Existing Stale Artifacts

`results\phase2_real_manifest\manifest.csv` was generated on 2026-07-07, before the BUG-003 fix.

Sample rows from the stale manifest:

| image_id | quality_score | landmark_score | landmark_out_ratio | landmark_bbox_area | landmark_center_dist |
|---|---:|---:|---:|---:|---:|
| 0 | 0.394719 | 0.000000 | 0.661765 | 0.000023 | 0.706658 |
| 1 | 0.358395 | 0.000000 | 0.735294 | 0.000025 | 0.706512 |
| 10 | 0.395272 | 0.000000 | 0.764706 | 0.000025 | 0.706583 |
| 100 | 0.392171 | 0.000000 | 0.808824 | 0.000027 | 0.706502 |

The old inference outputs reuse these stale quality values:

| Run | Count | Standardize | Weak Standardize | Reject |
|---|---:|---:|---:|---:|
| `phase2_real_infer_stage1_recovered` | 10,000 | 46 | 0 | 9,954 |
| `phase2_real_infer_stage2_recovered` | 10,000 | 34 | 2 | 9,964 |
| `phase2_real_infer_stage3_recovered` | 10,000 | 8 | 1 | 9,991 |

This confirms that the nearly-all-reject inference outputs are stale BUG-003-era artifacts, not valid post-fix Phase2 results.

## Current-Code Recompute Spot Check

Using the current fixed `phase2/features.py` on the same DECA `.mat` files:

| image_id | quality_score | landmark_score | landmark_out_ratio | landmark_bbox_area | landmark_center_dist |
|---|---:|---:|---:|---:|---:|
| 0 | 0.691230 | 0.926598 | 0.000000 | 0.292543 | 0.077116 |
| 1 | 0.657592 | 0.934991 | 0.000000 | 0.314696 | 0.083583 |
| 2 | 0.694587 | 0.938826 | 0.000000 | 0.301943 | 0.078653 |
| 3 | 0.716467 | 0.967526 | 0.000000 | 0.329860 | 0.041752 |
| 100 | 0.693544 | 0.941791 | 0.000000 | 0.342375 | 0.074840 |
| 9999 | 0.631113 | 0.896180 | 0.000000 | 0.279041 | 0.084978 |

The recomputed values match the documented post-fix spot-check range. The bug fix is active in code.

## Search for Corrected Output Version

A scan of manifest-like CSV files under the project found no full corrected Phase2 manifest or inference output.

Observed relevant manifests:

| File | Evidence |
|---|---|
| `results\phase2_real_manifest\manifest.csv` | `landmark_score` first 200 mean = 0.000000 |
| `results\screening_threshold_benchmark\screening_threshold_review_manifest.csv` | `landmark_score` first 200 mean = 0.000000 |
| `results\phase2_real_infer_stage1_recovered\phase2_inference_manifest.csv` | quality first 200 mean = 0.380998; decisions first 200 = all reject |
| `results\phase2_real_infer_stage2_recovered\phase2_inference_manifest.csv` | quality first 200 mean = 0.380998; decisions first 200 = all reject |
| `results\phase2_real_infer_stage3_recovered\phase2_inference_manifest.csv` | quality first 200 mean = 0.380998; decisions first 200 = all reject |

Directories modified after the BUG-003 fix that looked relevant:

- `results\screening_threshold_benchmark`
- `phase2`
- `phase2\__pycache__`

No directory resembling a full corrected `phase2_real_manifest`, `phase2_real_infer_*`, `quality_manifest`, `fixed`, `corrected`, `rebuild`, or `retrain` output was found.

## Conclusion

The code fix exists and works.

The recovered Phase2 10K manifest/inference outputs are stale pre-fix artifacts.

No full corrected 10K Phase2 output version was found on 5060 during this audit. The corrected work appears to have reached at least a 200-sample spot check, but not a persisted full regenerated manifest/inference directory, or it was saved outside the inspected project tree.

## Recommended Next Check

Before retraining anything, regenerate only the Phase2 manifest with fixed landmark scoring:

```powershell
python -m phase2.build_manifest `
  --deca-results-dir DECA\results\archive_phase2_params `
  --arcface-manifest results\phase1_parity\phase1_master_manifest.csv `
  --out-csv results\phase2_manifest_bug003_fixed\manifest.csv `
  --out-json results\phase2_manifest_bug003_fixed\manifest_summary.json
```

Then compare the new quality distribution against:

- old `results\phase2_real_manifest\manifest.csv`
- documented 200-sample post-fix metrics
- Phase1 p95/p97.5 labels

Only after this should Phase2 inference or retraining be rerun.
