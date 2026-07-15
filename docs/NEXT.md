# Next Actions

- **2026-07-15**: Phase 1 parity gap confirmed between the historical workstation and win-lenovo
- **Next**: Use the completed Phase 1 manifest for p95/p97.5 downstream benchmarking and rendering verification

### Immediate

1. Compare p95 and p97.5 using the joined DECA, gaze, and ArcFace features; keep p95 canonical until that benchmark is complete.
2. Verify standardized normal/displacement/render outputs by image ID.
3. Rebuild the Phase2 quality manifest using corrected landmarks plus ArcFace fields.
4. Retrain/evaluate Phase2 p95 and p97.5 branches with identity-consistency metrics.
5. Archive model/config hashes and compact result summaries outside the Lenovo-only disk.

### Acceptance Criteria

- Exactly 10,000 unique source image IDs are inventoried.
- Every row has a cleaning label and DECA status.
- L2CS coverage is 10,000 rows or every failure has an explicit reason.
- ArcFace strict/full coverage and retry provenance are reported separately.
- Render artifacts are linked by manifest paths; no silent filename reconstruction.
- Source image hashes are recorded so migrated and regenerated artifacts can be compared.

### Pending User Decisions

- [待確認] Target paper/conference venue
- [待确认] Need for additional dataset restoration
- [待确认] Backup strategy for checkpoints and data
- [待确认] Can screening_p95 and screening_p975 be archived? (23 GB duplicated data)
