# Downstream Condition Dataset Schema

Date: 2026-09-02

## Purpose

This schema defines the manifest format for future downstream image-generation or condition-map experiments. It is an interface document only. It does not claim that Phase2.1 Gate is deployment-qualified.

Each row represents one source face sample and its available conditioning artifacts.

For bounded audits or overfit experiments, pass `--include-ids-file`. The
builder preserves the ID-file order, rejects duplicates or IDs absent from the
Phase1 manifest, and records its count and SHA256 in `dataset_summary.json`.

## JSONL Row

```json
{
  "image_id": "sample_000001",
  "split": "train",
  "source_image": "path/to/source.jpg",
  "source_image_exists": true,
  "deca_mat": "path/to/deca.mat",
  "deca_mat_exists": true,
  "phase2_npz": "path/to/phase2_output.npz",
  "phase2_npz_exists": true,
  "source_depth_map": null,
  "source_normal_map": null,
  "source_landmark_map": null,
  "source_face_mask": null,
  "source_eye_mask": null,
  "target_depth_map": null,
  "target_normal_map": null,
  "target_landmark_map": null,
  "target_face_mask": null,
  "target_eye_mask": null,
  "target_gaze_heatmap": null,
  "depth_map": null,
  "normal_map": null,
  "landmark_map": null,
  "face_mask": null,
  "eye_mask": null,
  "gaze_heatmap": null,
  "modalities_todo": ["source_depth_map", "source_normal_map", "source_landmark_map", "source_face_mask", "source_eye_mask", "target_depth_map", "target_normal_map", "target_landmark_map", "target_face_mask", "target_eye_mask", "target_gaze_heatmap"],
  "arcface_embedding": "",
  "arcface_embedding_exists": false,
  "gaze_pitch": null,
  "gaze_yaw": null,
  "gaze_camera_x": null,
  "gaze_camera_y": null,
  "gaze_camera_z": null,
  "gaze_head_x": null,
  "gaze_head_y": null,
  "gaze_head_z": null,
  "target_gaze_head_x": null,
  "target_gaze_head_y": null,
  "target_gaze_head_z": null,
  "gaze_policy": "preserve_eye_in_head",
  "gaze_coordinate_status": "pending_head_rotation",
  "alpha_expression": null,
  "alpha_head_pose": null,
  "alpha_jaw_pose": null,
  "standardized_exp_norm": null,
  "standardized_head_pose_norm": null,
  "standardized_jaw_pose_norm": null,
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
| `source_image_exists` / `deca_mat_exists` / `phase2_npz_exists` | bool | builder | Explicit path validation |
| `source_depth_map` / `target_depth_map` | string/null | Phase3 DECA condition cache | Source reconstruction / standardized target depth |
| `source_normal_map` / `target_normal_map` | string/null | Phase3 DECA condition cache | Source reconstruction / standardized target normal |
| `source_landmark_map` / `target_landmark_map` | string/null | Phase3 DECA condition cache | Source reconstruction / standardized target landmarks |
| `source_face_mask` / `target_face_mask` | string/null | Phase3 DECA condition cache | Source/target face masks |
| `source_eye_mask` / `target_eye_mask` | string/null | Phase3 DECA condition cache | Source/target eye masks |
| `target_gaze_heatmap` | string/null | approved gaze coordinate stage | Eye-anchored target gaze direction; null before approval |
| `depth_map` / `normal_map` / `landmark_map` / `face_mask` / `eye_mask` / `gaze_heatmap` | string/null | builder | Backward-compatible aliases of the corresponding target fields |
| `modalities_todo` | list[string] | builder | Modalities absent from the supplied cache |
| `arcface_embedding` / `arcface_embedding_exists` | string / bool | ArcFace extractor | Validated identity condition path and existence flag |
| `gaze_pitch` | number/null | L2CS | Gaze diagnostic |
| `gaze_yaw` | number/null | L2CS | Gaze diagnostic |
| `gaze_camera_x/y/z` | number/null | L2CS | Visual axis in the L2CS camera frame |
| `gaze_head_x/y/z` | number/null | approved Phase3 condition cache | Eye-in-head gaze after coordinate transformation |
| `target_gaze_head_x/y/z` | number/null | approved Phase3 condition cache + policy | Eye-in-head target independent of target head pose |
| `gaze_policy` | string | experiment config | `preserve_eye_in_head`, `canonical_camera_gaze`, or `controlled_head_local` |
| `gaze_coordinate_status` | string | geometry stage | Whether camera/head coordinate conversion is complete |
| `alpha_expression` | number/null | Phase2 inference | Expression normalization strength |
| `alpha_head_pose` | number/null | Phase2 inference | Head-pose normalization strength |
| `alpha_jaw_pose` | number/null | Phase2 inference | Jaw-pose normalization strength |
| `standardized_exp_norm` | number/null | Phase2 inference | Standardized expression norm |
| `standardized_head_pose_norm` | number/null | Phase2 inference | Standardized head-pose norm |
| `standardized_jaw_pose_norm` | number/null | Phase2 inference | Standardized jaw-pose norm |
| `quality_score` | number/null | XGBoost/heuristic | Quality feature |
| `quality_label` | string | XGBoost/heuristic | Stratified analysis |
| `phase2_confidence` | number/null | Phase2 inference | Diagnostic feature |
| `phase2_reject_score` | number/null | Phase2 inference | Diagnostic feature |
| `phase2_gate_decision` | string | frozen gate, if available | Filtering audit |
| `rescue_source` | bool | rescue manifest | Must remain audit-only |

## Missing Values

- Missing source or computed artifact paths: `""`
- Future, not-yet-generated modality paths: `null` and the field name in `modalities_todo`
- Missing numeric values: `null`
- Missing labels or decisions: `""`
- Missing artifacts must be listed in `missing_fields`.
- Missing values must not be filled with `0`, because zero may be a valid pose, gaze, or score value.
- `gaze_head_*` must remain `null` until a validated head-to-camera rotation is available; camera-frame L2CS angles must not be relabeled as eye-in-head gaze.
- The builder accepts `--gaze-policy preserve_eye_in_head|canonical_camera_gaze|controlled_head_local`; the selected value is saved in every row and in `dataset_summary.json`.

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
