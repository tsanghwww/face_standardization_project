# Condition Dataset Schema

## Purpose

This document defines the unified JSONL sample format for downstream
image-generation experiments. It is an interface contract only: it does not
start diffusion or ControlNet training, and it does not require Phase2 to be
fully closed before the manifest skeleton can be built.

Each line in `train.jsonl`, `val.jsonl`, and `test.jsonl` should be a JSON
object with this shape:

```json
{
  "image_id": "0001",
  "source_image": "...",
  "deca_mat": "...",
  "phase2_npz": "...",
  "depth_map": "...",
  "normal_map": "...",
  "landmark_map": "...",
  "arcface_embedding": "...",
  "gaze_pitch": 0.0,
  "gaze_yaw": 0.0,
  "quality_score": 0.0,
  "phase2_confidence": 0.0,
  "phase2_reject_score": 0.0,
  "split": "train"
}
```

## Field Contract

| Field | Required | Source | Phase2-dependent | Missing value | Primary use |
|---|---|---|---|---|---|
| `image_id` | Yes | Phase1 master manifest | No | Row should be rejected | Join key, training, evaluation, debugging |
| `source_image` | Yes | Phase1 source-image path | No | `null` if unavailable in a legacy manifest | Training input, identity checks, debugging |
| `deca_mat` | Yes | DECA extraction artifact | No | `null` if DECA has not run | Geometry conditions, pose evaluation |
| `phase2_npz` | Optional until Phase2 is complete | Phase2 inference manifest | Yes | `null` | Standardized parameter condition, Phase2 debugging |
| `depth_map` | Optional | Future DECA/renderer condition export | No | `null` | ControlNet-style spatial condition |
| `normal_map` | Optional | Future DECA/renderer condition export | No | `null` | ControlNet-style spatial condition |
| `landmark_map` | Optional | Future landmark rasterization export | No | `null` | ControlNet-style spatial condition |
| `arcface_embedding` | Optional but recommended | ArcFace extraction artifact | No | `null` | Identity conditioning and identity evaluation |
| `gaze_pitch` | Optional | L2CS gaze manifest or Phase1 merged manifest | No | `null` | Gaze-behavior evaluation or future gaze condition |
| `gaze_yaw` | Optional | L2CS gaze manifest or Phase1 merged manifest | No | `null` | Gaze-behavior evaluation or future gaze condition |
| `quality_score` | Optional | Phase1 screening or quality model output | No | `null` | Filtering, sampling, debugging |
| `phase2_confidence` | Optional until Phase2 is stable | Phase2 inference manifest | Yes | `null` | Confidence-aware sampling, evaluation stratification |
| `phase2_reject_score` | Optional until Phase2 is stable | Phase2 inference manifest | Yes | `null` | Rejection filtering and debugging |
| `split` | Yes | Split files or manifest split column | No | Default to `train` only for skeleton builds | Dataset partitioning |

## Required Fields

The minimum usable downstream row is:

- `image_id`
- `source_image`
- `deca_mat`
- `split`

Rows missing `image_id` should not be emitted. Other required fields may be
temporarily emitted as `null` by skeleton tooling, but the summary file must
count those missing values so they are visible before training.

## Optional Fields

Optional fields are allowed to be `null` until their upstream artifact exists.
This includes Phase2 outputs, condition maps, ArcFace embeddings, gaze labels,
and quality scores. Optional does not mean unimportant; it means the downstream
manifest can be prepared before every artifact is ready.

## Missing Values

Use JSON `null` for unknown, not-yet-generated, or intentionally unavailable
fields. Do not use empty strings, `"NA"`, `"missing"`, or fake zero values.
Numeric zeros should only be used when zero is a real measurement.

## Split Semantics

Valid split values are:

- `train`
- `val`
- `test`

The dataset builder may read split assignments from `train.txt`, `val.txt`, and
`test.txt` in a split directory. If no split assignment exists, the skeleton may
fall back to a manifest `split` column, then to `train`.

## Debug Fields

Builder-generated debug fields such as `debug_path_checks` may be included in
early skeleton manifests. They are for validation only and should not be used as
model inputs.
