# Downstream Ablation Plan

Date: 2026-09-01

## Purpose

This plan defines future downstream image-generation ablations. It is a planning artifact only; full diffusion or ControlNet training has not started.

## Experiment Groups

| Group | Description | Main Input | Hypothesis | Phase2 Dependency |
| --- | --- | --- | --- | --- |
| A | Hard-zero parameter baseline | hard-zero DECA parameters | Direct neutralization is simple but may damage identity | Low |
| B | Phase2-only parameter standardization | Phase2 standardized parameters | Learned parameter standardization is safer than hard-zero | High |
| C | Image U-Net baseline | source image only | Image-only model may ignore explicit 3D structure | Low |
| D | ControlNet with landmarks only | landmark map | Sparse geometry helps alignment | Medium |
| E | ControlNet with depth only | depth map | Dense geometry improves pose consistency | Medium |
| F | ControlNet with normals only | normal map | Surface orientation improves face structure | Medium |
| G | ControlNet with DECA vector | DECA vector | Parametric 3D features improve controllability | Medium |
| H | ControlNet with Phase2 vector | Phase2 standardized vector | Standardized parameters improve canonical output | High |
| I | Full model with identity condition | Phase2 vector + ArcFace | Identity condition reduces identity drift | High |
| J | Coupled camera-gaze condition | Phase2 vector + ArcFace + L2CS camera gaze | Camera-frame conditioning is insufficient when head pose changes | High |
| K | Disentangled head-local gaze preservation | Separate head target + eye-in-head gaze target + ArcFace | Head pose can be standardized while preserving eye-in-head gaze | High |
| L | Disentangled gaze control | Fixed head target + sampled eye-in-head gaze target + ArcFace | Gaze can change without changing head pose or identity | High |

## Metrics

| Metric | Tool | Purpose |
| --- | --- | --- |
| ArcFace cosine | ArcFace | Identity preservation |
| DECA/L2CS pose delta | DECA/L2CS | Head-pose standardization |
| Camera-frame gaze delta | L2CS | Diagnostic expected to vary with head rotation |
| Head-local gaze angular error | L2CS + head rotation | Eye-in-head preservation or target accuracy |
| Head-to-gaze leakage | intervention pairs | Gaze change caused by head-only control |
| Gaze-to-head leakage | intervention pairs | Head change caused by gaze-only control |
| Render/generation failure rate | pipeline status | Robustness |
| Coverage | manifest accounting | How many inputs remain usable |
| Stratified performance | quality labels/source groups | Robustness across sample difficulty |

## Required Guardrails

- Do not use rescue outputs as primary training samples.
- Do not tune thresholds on fixed test.
- Always report coverage alongside quality metrics.
- Keep identity, pose, and gaze metrics separate.
- Require head-only and gaze-only interventions; ordinary reconstruction metrics cannot demonstrate disentanglement.
- Do not select losses or thresholds on the 775 fixed test.
- Treat Phase2.1 Gate as diagnostic unless a future calibration split qualifies it.

## First Valid Ablation

The first valid downstream ablation should be manifest-only:

1. Build JSONL rows from existing Phase1/Phase2 manifests.
2. Verify missing-field accounting.
3. Run placeholder evaluators and confirm output schemas.
4. Freeze split files before any model training.

## First Gaze-Disentanglement Ablation

After coordinate validation, compare J versus K on a frozen validation subset. For each source, render reconstruction and head-only intervention outputs. K is supported only if it reduces head-local gaze leakage without worsening head target error, identity cosine, failure rate, or coverage. Group L begins only after the gaze-only target interface is implemented.
