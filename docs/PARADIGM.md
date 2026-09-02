# Project Paradigm

Analysis date: 2026-08-19

Scope: This document defines the working paradigm for the project after checking both the local Mac workspace and the `win-lenovo` artifacts under `D:\face_standardization_project`.

## 1. One-Sentence Paradigm

This project should be treated as a manifest-centered, DECA/FLAME-anchored, quality-aware 3D parameter-space face standardization system, with image-level diffusion/control generation reserved as a downstream extension rather than the current core.

## 2. Current Effective Paradigm

The current effective paradigm is:

```text
Raw image
  -> DECA / FLAME structured representation
  -> Phase1 auxiliary identity, gaze, cleaning, and quality signals
  -> XGBoost / heuristic quality scoring
  -> Phase2 learned partial standardization in expression and pose parameter space
  -> baseline comparison, rendering validation, identity/gaze/pose evaluation
  -> optional future image-level generator
```

The important choice is representation-first rather than image-first. The project does not yet primarily learn to generate standardized RGB faces. It first learns to manipulate structured 3D face parameters in a controlled and auditable way.

## 3. Core Research Object

The current research object is the Phase2 `ConditionGenerator`.

It predicts:

- canonical targets for expression, head pose, and jaw pose;
- per-sample standardization strengths (`alpha_expression`, `alpha_head_pose`, `alpha_jaw_pose`);
- confidence;
- reject score;
- standardized expression and pose outputs.

The scientific claim should therefore be framed around quality-aware partial canonicalization in DECA/FLAME parameter space, not around full image generation.

## 4. What Is Already Real

The following have been verified as concrete project artifacts or code paths:

| Area | Current state |
|---|---|
| Raw dataset | 10,000 StyleGAN2 images verified on `win-lenovo` |
| DECA parameters | 10,000 `.mat` outputs verified on `win-lenovo` |
| Phase1 master manifest | 10,000 rows verified at `results/phase1_parity/phase1_master_manifest.csv` |
| L2CS gaze extraction | 10,000/10,000 documented complete |
| ArcFace extraction | 9,990/10,000 documented complete |
| BUG-003 landmark coordinate fix | Implemented and reflected in corrected Phase2 manifest |
| ArcFace schema fix | Implemented in remote `phase2/features.py`; ArcFace score/flags reach Phase2 features |
| XGBoost quality model | Trained on p95 labels; validation AUC 0.900 |
| Phase2 retraining | Stage 1/2/3 corrected runs complete on `win-lenovo` |
| Phase2 full inference | 10,000 outputs; 9,999 standardize, 1 reject |
| Hard-zero comparison | Parameter-space comparison complete on same 10,000 samples |
| Single-image visualization | `IMG_6033` visual comparison exists; note says 2D warp is not diffusion output |

## 5. Current Best Evidence

The strongest current evidence supports a parameter-space claim:

| Metric | Corrected Phase2 result |
|---|---:|
| Stage 1 best val loss | 0.1411 |
| Stage 2 best val loss | 0.1452 |
| Stage 3 best val loss | 0.1495 |
| Full inference standardize count | 9,999 / 10,000 |
| Full inference reject count | 1 / 10,000 |
| Expression norm after Phase2 | 1.7% of original mean |
| Head pose norm after Phase2 | 2.1% of original mean |
| Jaw pose norm after Phase2 | 11.1% of original mean |
| XGBoost quality validation AUC | 0.900 |

This evidence says the learned Phase2 model can strongly reduce expression/head-pose/jaw-pose magnitude in parameter space while retaining per-sample adaptivity.

It does not yet prove image-level identity preservation, perceptual quality, gaze disentanglement, or diffusion-based frontalization.

## 6. Non-Negotiable Boundaries

The project should keep these boundaries explicit:

1. Parameter-space standardization is not the same as image-level face generation.
2. Head pose normalization is not the same as eye-gaze disentanglement.
3. ArcFace/L2CS extraction provides measurement signals, not proof of identity or gaze preservation unless used in a controlled evaluation.
4. A single visual example is not a full rendered-output validation.
5. XGBoost quality scoring is a proxy quality gate, not a ground-truth visual-quality oracle.
6. StyleGAN2-only data cannot support broad in-the-wild claims without external validation.

## 7. Recommended Claim Hierarchy

Use a staged claim hierarchy:

### Claim A: Supported Now

Quality-aware learned partial standardization can reduce DECA/FLAME expression and pose residuals more adaptively than hard-zeroing.

### Claim B: Near-Term, Needs Evaluation

The learned parameter-space standardizer improves rendered geometry stability and preserves identity better than hard-zeroing.

Required next evidence:

- full or representative rendered-output validation;
- ArcFace identity similarity on rendered outputs;
- failure/collapse manifest;
- held-out split reporting.

### Claim C: Future Extension

Standardized DECA/FLAME conditions can serve as inputs to a downstream image-level diffusion or ControlNet-style generator.

Required next evidence:

- condition dataset manifest;
- image-like conditions such as landmarks, normals, depth, masks;
- generation model code;
- generated outputs;
- image quality and identity/gaze evaluation.

### Claim D: Not Yet Supported

The claim that the project has achieved gaze disentanglement is not yet supported.

Required next evidence:

- explicit gaze objective or control protocol;
- reliable gaze labels or validated pseudo-label protocol;
- metrics separating eye gaze from head pose;
- ablation proving gaze changes do not leak into identity or head pose.

## 8. Practical Research Direction

The strongest next direction is not to start diffusion training immediately. The best next direction is:

1. Freeze the corrected Phase2 run as the canonical parameter-space baseline.
2. Define reproducible train/validation/test splits.
3. Run full or representative rendered validation for original, hard-zero, and Phase2 outputs.
4. Evaluate identity preservation, pose canonicalization, gaze behavior, and collapse/failure rate.
5. Run ablations for XGBoost quality, alpha blending, augmentation, reject gate, and stage selection.
6. Decide whether the first paper is a Phase2 parameter-space paper or whether it waits for downstream image generation.

## 9. Paradigm Summary

The project should be narrated as a controlled 3D standardization pipeline:

```text
Data provenance first.
3D parameter representation second.
Quality-aware partial control third.
Image-level validation fourth.
Diffusion generation fifth.
Gaze disentanglement only after explicit evidence.
```

This keeps the project honest, defensible, and extensible.
