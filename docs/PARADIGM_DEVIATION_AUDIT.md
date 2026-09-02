# Paradigm Deviation Audit

Analysis date: 2026-08-19

Scope: This audit compares the project paradigm described in local documents with the actual artifacts verified on `win-lenovo`. It separates direction-level deviations from ordinary unfinished work.

## 1. Classification Rules

### Direction-Level Deviation

A direction-level deviation changes, inflates, or confuses the research claim. These are dangerous because they can make the project appear to prove something it has not tested.

Examples:

- claiming latent diffusion without a diffusion model;
- claiming gaze disentanglement from L2CS extraction alone;
- treating parameter norm reduction as image-level face quality;
- using a single visualization as full validation.

### Work Missing

A work-missing item is aligned with the paradigm but incomplete. These are normal project gaps. They do not require changing direction; they require execution, evaluation, or documentation.

Examples:

- full rendered validation not yet run;
- identity metrics not yet computed;
- ablation suite not yet complete;
- local Mac mirror missing large remote artifacts.

## 2. Current Reality After Remote Check

The local status documents from 2026-08-07 are partly stale. The remote `win-lenovo` machine now contains corrected 2026-08-08 Phase2 artifacts:

| Artifact | Status |
|---|---|
| `archive/generated_yellow-stylegan2` | 10,000 source images verified |
| `DECA/results/archive_phase2_params` | 10,000 `.mat` outputs verified |
| `results/phase1_parity/phase1_master_manifest.csv` | 10,000 rows verified |
| `results/phase2_manifest_bug003_fixed_arcface_ok` | corrected Phase2 manifest exists |
| `results/phase2_xgb_quality_bug003_fixed_arcface_ok` | XGBoost quality model exists; val AUC 0.900 |
| `results/phase2_train_stage{1,2,3}_bug003_fixed_arcface_ok` | corrected Stage 1/2/3 checkpoints exist |
| `results/phase2_infer_sanity_bug003_fixed_arcface_ok` | 10,000 inference outputs; 9,999 standardize, 1 reject |
| `results/phase2_compare_bug003_fixed_arcface_ok` | hard-zero vs Phase2 parameter comparison exists |
| `results/single_test_phase2_img6033/visual_comparison_final` | one visual comparison exists |

Therefore, the corrected Phase2 closed-loop is no longer missing. The remaining core gap is evaluation beyond parameter-space metrics.

## 3. Direction-Level Deviations

### D1. Project Title Overclaims the Implemented System

Type: Direction-level deviation

Document signal:

- The full title frames the project as `3D fusion control + latent-space diffusion + gaze disentanglement`.

Actual state:

- The implemented system is a DECA/FLAME parameter-space standardizer.
- No Stable Diffusion, ControlNet, VAE/UNet training loop, sampler, or generated face pipeline exists.
- The single `IMG_6033` output explicitly notes that the 2D warp is a visualization preview, not diffusion output.

Why it matters:

- This is the largest claim-scope mismatch. If kept as the main claim, the project will be judged against generative-model and gaze-control standards it does not yet satisfy.

Recommended handling:

- Treat diffusion/control generation as a downstream extension.
- For the current paper/project milestone, use a title closer to quality-aware 3D parameter-space standardization.

### D2. Gaze Disentanglement Is Framed Ahead of Evidence

Type: Direction-level deviation

Document signal:

- Several planning documents mention gaze disentanglement as part of the broader goal.

Actual state:

- L2CS gaze extraction exists and is useful.
- There is no explicit gaze-control model, disentanglement loss, paired gaze protocol, or ground-truth gaze benchmark.

Why it matters:

- Measuring gaze is not the same as disentangling gaze.
- DECA head pose and L2CS eye gaze must remain separate signals.

Recommended handling:

- Say "gaze behavior evaluation" for current work.
- Reserve "gaze disentanglement" for a future branch with controlled evidence.

### D3. Parameter-Space Success Could Be Mistaken for Image-Level Success

Type: Direction-level deviation

Document signal:

- Some narrative language moves from standardized parameters to standardized faces.

Actual state:

- Corrected Phase2 strongly reduces expression and head-pose norms.
- Full rendered-output validation, image identity preservation, and perceptual quality evaluation are not complete.

Why it matters:

- A low parameter norm can still render poorly or damage identity.
- Hard-zero has perfect parameter residuals but may be visually worse, which shows why parameter metrics alone are insufficient.

Recommended handling:

- Keep "parameter-space standardization" as the supported claim.
- Require rendered validation before claiming face-standardization quality.

### D4. Single-Image Visualization Risks Becoming Anecdotal Evidence

Type: Direction-level deviation

Document signal:

- `single_test_phase2_img6033` contains useful visual outputs.

Actual state:

- It is one manually cropped sample.
- It has `arcface_status=0` in the single-run summary.
- It is not a full evaluation split.

Why it matters:

- A single image can guide debugging but cannot support claims about robustness, identity preservation, or collapse rate.

Recommended handling:

- Use it as a qualitative sanity check only.
- Build a manifest-backed rendered evaluation over representative samples.

### D5. Downstream Diffusion Skeletons Could Pull Attention Before Evaluation

Type: Direction-level deviation risk

Document signal:

- `task.md` asks for condition dataset and evaluation skeletons for downstream diffusion/ControlNet-style models.

Actual state:

- This is compatible with the paradigm if kept as preparation.
- It becomes a deviation only if it shifts work away from validating Phase2.

Why it matters:

- The project can become architecture-forward before the central Phase2 claim is measured.

Recommended handling:

- Keep downstream schema/design lightweight.
- Do not train diffusion or add large generator code until rendered validation and ablations are complete.

## 4. Work Missing, Not Direction Deviations

### M1. Full Rendered Output Validation

Type: Work missing

Why it is aligned:

- Rendering validation is exactly the next stage after parameter-space Phase2.

Current evidence:

- Single-image visual comparison exists.
- Full render parity, failure manifest, and representative panels are missing.

Needed work:

- Render original, hard-zero, and Phase2 outputs for a fixed subset or full 10K.
- Record render success/failure by `image_id`.
- Produce comparison grids and collapse/failure statistics.

### M2. ArcFace Identity Preservation Evaluation

Type: Work missing

Why it is aligned:

- Identity preservation is a core evaluation dimension for face standardization.

Current evidence:

- ArcFace extraction exists for source images.
- ArcFace features now correctly enter Phase2 quality features.
- ArcFace has not yet been used to compare source vs rendered/standardized outputs.

Needed work:

- Run ArcFace on rendered or generated outputs.
- Compare source and output embeddings.
- Report cosine similarity, failure rate, and examples.

### M3. Pose and Gaze Behavior Evaluation

Type: Work missing

Why it is aligned:

- Pose standardization and gaze behavior are evaluation targets.

Current evidence:

- DECA pose parameters exist.
- L2CS gaze summaries exist.
- Full post-standardization gaze behavior metrics do not yet exist.

Needed work:

- Compute post-output head-pose metrics.
- Run or reuse L2CS for output images when image outputs exist.
- Report gaze behavior descriptively without claiming disentanglement.

### M4. Ablation Suite

Type: Work missing

Why it is aligned:

- Ablations test whether each component of the current paradigm matters.

Current evidence:

- Hard-zero vs Phase2 comparison exists.
- No full ablation suite for XGBoost, alpha blending, augmentation, reject gate, stage selection, or p95/p97.5 branch exists.

Needed work:

- Define fixed splits and metrics.
- Run no-XGBoost, no-augmentation, hard-zero, stage1/2/3, no-reject, and p95/p97.5 comparisons.

### M5. Train/Validation/Test Split Definition

Type: Work missing

Why it is aligned:

- Reproducible claims need split boundaries.

Current evidence:

- Training scripts use random train/val splits.
- A paper-grade held-out split registry is not clearly present.

Needed work:

- Create stable split files by `image_id`.
- Store split generation seed and exclusion rules.
- Avoid threshold calibration on final test IDs.

### M6. Real-Image or Multi-Identity Validation

Type: Work missing

Why it is aligned:

- External validity is needed for broader claims.

Current evidence:

- The verified 10K dataset is StyleGAN2 synthetic.
- Current audit notes mention limited real-image coverage and possible single-identity/data-scope limitations.

Needed work:

- Add a real-image validation subset or clearly limit the claim to the synthetic dataset.
- If multi-identity support is required, verify identity diversity and labels.

### M7. Local/Remote Artifact Synchronization

Type: Work missing

Why it is aligned:

- The paradigm is manifest-centered and reproducibility-focused.

Current evidence:

- Corrected artifacts exist on `win-lenovo`.
- The Mac workspace does not contain full data/results/checkpoints.
- Some local status docs are stale relative to remote results.

Needed work:

- Update status docs to mention the 2026-08-08 corrected runs.
- Decide which summaries should be mirrored locally.
- Keep large outputs remote or external, but mirror small summaries and manifests when allowed.

### M8. Paper-Ready Claim and Contribution Statement

Type: Work missing

Why it is aligned:

- The current system needs a clear paper boundary.

Current evidence:

- The code supports a Phase2 parameter-space contribution.
- The broader title still includes diffusion and gaze disentanglement.

Needed work:

- Write a one-page research brief.
- Choose between a narrower Phase2 paper and a later full image-generation paper.
- Map each claim to required evidence.

## 5. Stale Documentation, Not Paradigm Failure

Several local documents are now behind the remote state:

| Document | Stale point |
|---|---|
| `CURRENT_STATUS.md` | says corrected full 10K Phase2 inference was not found; remote now has `phase2_infer_sanity_bug003_fixed_arcface_ok` |
| `PROJECT_RESTART_SUMMARY.md` | says no full corrected 10K Phase2 manifest/inference was found; remote now has corrected manifest, training, inference, and comparison |
| `STAGE_GAP_ANALYSIS.md` | still lists post-BUG-003 retrain confirmation as missing |
| `RECOVERY_EXPERIMENT_PLAN.md` | recovery experiments 4 and 5 are now partly complete on `win-lenovo` |

This is not a research direction deviation. It is a documentation synchronization gap.

## 6. Summary Table

| Item | Classification | Severity | Action |
|---|---|---:|---|
| Full title implies diffusion/gaze disentanglement | Direction deviation | High | Narrow current claim or mark future work |
| Gaze disentanglement language before control evidence | Direction deviation | High | Rename to gaze behavior evaluation |
| Parameter metrics treated as face-quality proof | Direction deviation | High | Require rendered identity/quality validation |
| Single visual example used as broad evidence | Direction deviation | Medium | Keep as sanity check only |
| Downstream diffusion skeletons before evaluation | Direction risk | Medium | Keep lightweight; do not train yet |
| Full rendered validation missing | Work missing | High | Run fixed-subset/full render evaluation |
| ArcFace identity output evaluation missing | Work missing | High | Compute source-output embedding similarity |
| Pose/gaze output metrics missing | Work missing | High | Add pose/gaze behavior reports |
| Ablation suite missing | Work missing | High | Run fixed split ablations |
| Stable split registry missing | Work missing | High | Create split manifests |
| Real-image/multi-identity validation missing | Work missing | Medium | Add external validation or narrow claims |
| Local docs stale vs remote results | Work missing | Medium | Update small docs/summaries |

## 7. Bottom Line

The project has not drifted away from its strongest feasible paradigm. In fact, the corrected `win-lenovo` artifacts show that the Phase2 parameter-space paradigm is now much stronger than the older local documents suggested.

The real danger is not that Phase2 is weak. The danger is overextending the story before evaluation catches up:

```text
Supported: quality-aware DECA/FLAME parameter-space standardization.
Nearly testable: rendered identity-preserving face standardization.
Future: diffusion/control generation.
Not yet supported: gaze disentanglement.
```

Treat the first line as the current project center. Treat the others as staged extensions with explicit evidence gates.
