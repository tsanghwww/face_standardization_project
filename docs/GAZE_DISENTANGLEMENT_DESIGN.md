# Gaze Disentanglement Design

Date: 2026-09-01

## Research Decision

Phase3 will explicitly target gaze disentanglement. The operational goal is to standardize head pose while independently preserving or controlling eye-in-head gaze. L2CS camera-frame gaze change alone is no longer the primary gaze metric.

This is a target and evaluation protocol, not a claim that the current model has already achieved disentanglement.

## Variables and Coordinate Frames

Let:

- $R_h \in SO(3)$ be the rotation from the head coordinate system to the camera coordinate system.
- $\mathbf{g}_{cam}$ be the unit gaze vector estimated in the camera coordinate system.
- $\mathbf{g}_{head}$ be eye-in-head gaze, expressed in the head coordinate system.

The decomposition is:

$$
\mathbf{g}_{head} = R_h^\top \mathbf{g}_{cam},
\qquad
\mathbf{g}_{cam} = R_h \mathbf{g}_{head}.
$$

If the downstream model changes head orientation to $R_h^*$ while preserving eye-in-head gaze, the expected camera-frame gaze is:

$$
\mathbf{g}_{cam}^* = R_h^* \mathbf{g}_{head}^{src}.
$$

This distinction matters because preserving $\mathbf{g}_{cam}$ while rotating the head generally changes the eyes relative to the head. The default Phase3 policy is therefore `preserve_eye_in_head`, not `preserve_camera_gaze`.

The project must verify the DECA/L2CS axis signs, handedness, crop transform, and whether DECA global pose maps head-to-camera before producing $\mathbf{g}_{head}$. Until that convention test passes, head-local fields remain `null`.

## Training Interface

The downstream condition must expose separate factors:

$$
z = [z_{id}, z_{shape}, z_{expr}, z_{head}, z_{gaze}, z_{quality}],
$$

where $z_{head}$ receives the target head rotation and $z_{gaze}$ receives the target eye-in-head gaze. They must not be concatenated into one unlabeled pose vector without factor-specific heads or controls.

For a generated output $\hat{x}$, the proposed objective is:

$$
\mathcal{L} =
\lambda_{id}\mathcal{L}_{id}
+ \lambda_h\mathcal{L}_{head}
+ \lambda_g\mathcal{L}_{gaze\_head}
+ \lambda_{h\rightarrow g}\mathcal{L}_{h\rightarrow g}
+ \lambda_{g\rightarrow h}\mathcal{L}_{g\rightarrow h}
+ \lambda_q\mathcal{L}_{quality}.
$$

The factor losses are:

$$
\mathcal{L}_{id}=1-\cos(e_{src},e_{out}),
$$

$$
\mathcal{L}_{head}=d_R(R_h^{out},R_h^*),
$$

$$
\mathcal{L}_{gaze\_head}=d_\angle(\mathbf{g}_{head}^{out},\mathbf{g}_{head}^*).
$$

$\mathcal{L}_{h\rightarrow g}$ penalizes changes in head-local gaze when only head pose is intervened on. $\mathcal{L}_{g\rightarrow h}$ penalizes head-pose changes when only the gaze target is intervened on.

## Required Intervention Evaluation

Every evaluated identity should, where technically possible, produce a small factorial set:

| Intervention | Head target | Eye-in-head target | Required invariance |
| --- | --- | --- | --- |
| Reconstruction | source | source | identity, head, gaze |
| Head-only | canonical or sampled | source | eye-in-head gaze |
| Gaze-only | source | canonical or sampled | head pose and identity |
| Joint | canonical or sampled | canonical or sampled | identity and target accuracy |

The primary disentanglement metrics are:

- Head target angular error.
- Eye-in-head gaze angular error.
- $h\rightarrow g$ leakage: eye-in-head gaze change under a head-only intervention.
- $g\rightarrow h$ leakage: head-pose change under a gaze-only intervention.
- ArcFace identity cosine and generation failure rate for every intervention.
- Coverage, with failures retained in the full split denominator.

Camera-frame gaze delta remains a diagnostic, because it is expected to change during head-only canonicalization even when eye-in-head gaze is perfectly preserved.

## Evidence Levels

| Level | Evidence | Permitted wording |
| --- | --- | --- |
| E0 | L2CS extraction only | Gaze measurement available |
| E1 | Camera/head coordinate conversion passes convention tests | Head-local gaze proxy available |
| E2 | Head-only and gaze-only intervention metrics on frozen validation | Gaze-disentanglement behavior evaluated |
| E3 | Independent test with paired or ground-truth gaze labels | Model demonstrates gaze/head-pose disentanglement within the tested domain |

The current project is between E0 and E1. It must not use E2 or E3 wording yet.

## Data Strategy Under Current Constraints

The project does not need to make a new large dataset a Phase3 entry condition. It can begin with L2CS pseudo-labels and DECA head rotation as weak supervision, while reporting their limitations. A small controlled or public gaze set can later be reserved for validation rather than used to expand the main training set.

ETH-XGaze is a suitable future validation source because it was designed with large head-pose and gaze variation and calibrated gaze targets. It is not required for the first engineering smoke.

## Supporting Papers

- [L2CS-Net](https://arxiv.org/abs/2203.03339) supports unconstrained camera-frame gaze estimation, but does not by itself establish factor disentanglement.
- [Few-Shot Adaptive Gaze Estimation / DT-ED](https://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf) uses separate rotation-aware gaze and head-pose embeddings.
- [ST-ED](https://ait.ethz.ch/sted-gaze) demonstrates separate control and evaluates gaze/head redirection disentanglement.
- [Fine Gaze Redirection Learning](https://openaccess.thecvf.com/content/WACV2023/papers/Park_Fine_Gaze_Redirection_Learning_With_Gaze_Hardness-Aware_Transformation_WACV_2023_paper.pdf) evaluates bidirectional factor influence by perturbing gaze and head factors separately.
- [ETH-XGaze](https://ait.ethz.ch/xgaze) provides calibrated gaze targets under extreme head-pose and gaze variation.

## Immediate Engineering Gate

Before generator training:

1. Validate coordinate conventions on synthetic rotations and manually inspected samples.
2. Build source $R_h$, $\mathbf{g}_{cam}$, and $\mathbf{g}_{head}$ manifests with explicit failure status.
3. Extend the evaluator to consume output DECA/L2CS measurements.
4. Freeze head-only and gaze-only intervention definitions and validation IDs.
5. Keep rescue samples audit-only and keep the 775 fixed test isolated from threshold and loss-weight selection.
