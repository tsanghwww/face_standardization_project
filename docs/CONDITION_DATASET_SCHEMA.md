# Downstream Condition Dataset Schema

Date: 2026-08-31

## Purpose

This schema defines the manifest format for future downstream image-generation or condition-map experiments. It is an interface document only. It does not claim that Phase2.1 Gate is deployment-qualified.

Each row represents one source face sample and its available conditioning artifacts.

## JSONL Row

```json
{
  "image_id": "sample_000001",
  "split": "train",
  "source_image": "path/to/source.jpg",
  "deca_mat": "path/to/deca.mat",
  "phase2_npz": "path/to/phase2_output.npz",
  "depth_map": "",
  "normal_map": "",
  "landmark_map": "",
  "face_mask": "",
  "arcface_embedding": "",
  "gaze_pitch": null,
  "gaze_yaw": null,
  "quality_score": null,
  "quality_label": "",
  "phase2_confidence": null,
  "phase2_reject_score": null,
  "phase2_gate_decision": "",
  "rescue_source": false,
  "status": "available",
  "missing_fields": []
}
```

## Required Fields

| Field | Type | Source | Use |
| --- | --- | --- | --- |
| `image_id` | string | Phase1/Phase2 manifest | Join key |
| `split` | string | split file or manifest | Train/val/test routing |
| `source_image` | string | Phase1/base manifest | Image input |
| `deca_mat` | string | DECA extraction | 3D parameter conditioning |
| `status` | string | builder | Availability and failure audit |
| `missing_fields` | list[string] | builder | Explicit missing-data record |

Required fields must be present in every JSONL row. Missing paths should be represented as empty strings and listed in `missing_fields`.

## Optional Fields

| Field | Type | Source | Use |
| --- | --- | --- | --- |
| `phase2_npz` | string | Phase2 inference | Standardized parameter conditioning |
| `depth_map` | string | future renderer | ControlNet/image condition |
| `normal_map` | string | future renderer | ControlNet/image condition |
| `landmark_map` | string | future renderer | ControlNet/image condition |
| `face_mask` | string | future parser/renderer | Loss mask or conditioning |
| `arcface_embedding` | string | ArcFace extractor | Identity condition or evaluation |
| `gaze_pitch` | number/null | L2CS | Gaze diagnostic |
| `gaze_yaw` | number/null | L2CS | Gaze diagnostic |
| `quality_score` | number/null | XGBoost/heuristic | Quality feature |
| `quality_label` | string | XGBoost/heuristic | Stratified analysis |
| `phase2_confidence` | number/null | Phase2 inference | Diagnostic feature |
| `phase2_reject_score` | number/null | Phase2 inference | Diagnostic feature |
| `phase2_gate_decision` | string | frozen gate, if available | Filtering audit |
| `rescue_source` | bool | rescue manifest | Must remain audit-only |

## Missing Values

- Missing paths: `""`
- Missing numeric values: `null`
- Missing labels or decisions: `""`
- Missing artifacts must be listed in `missing_fields`.
- Missing values must not be filled with `0`, because zero may be a valid pose, gaze, or score value.

## Split Rules

- The 775 fixed test set must remain isolated.
- Gate thresholds must not be tuned on fixed test.
- Rescue outputs must not be silently mixed into the primary path.
- Downstream exploratory training may use Phase2 outputs only after recording the exact Phase2 commit, checkpoint, manifest, and filtering rule.

## Status Values

| Status | Meaning |
| --- | --- |
| `available` | Required source image and DECA artifacts exist |
| `missing_source` | Source image path is missing or absent |
| `missing_deca` | DECA mat path is missing or absent |
| `phase2_missing` | Phase2 output is absent, but base row is still auditable |
| `excluded` | Row is explicitly excluded by split or gate policy |
