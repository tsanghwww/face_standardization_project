# Condition Dataset Schema


## 1. Purpose

This document defines the unified sample format for the downstream
image generation model.

Each sample contains the original face image and corresponding
conditions from DECA, Phase2, ArcFace, and gaze estimation modules.


## 2. Sample Format

Each sample is represented as a JSON object:


```json
{
  "image_id": "0001",
  "source_image": "path/to/image.jpg",
  "deca_mat": "path/to/deca.mat",
  "phase2_npz": "path/to/phase2.npz",
  "depth_map": "path/to/depth.png",
  "normal_map": "path/to/normal.png",
  "landmark_map": "path/to/landmark.png",
  "arcface_embedding": "path/to/embedding.npy",
  "gaze_pitch": 0.0,
  "gaze_yaw": 0.0,
  "quality_score": 0.0,
  "phase2_confidence": 0.0,
  "phase2_reject_score": 0.0,
  "split": "train"
}