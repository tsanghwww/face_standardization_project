# Recovery Workflow

## Incident: 2060 Workstation Loss (2026-07)

### What Happened
- Original RTX 2060 workstation completely lost
- All project data on that machine irretrievable

### Recovery Actions Taken

1. **Mac**: Transferred remaining local files
2. **External backup**: Restored to 5060 workstation (win-lenovo)
3. **Code**: Recovered via external backup → GitHub sync
4. **Model weights**: DECA pretrained model and FLAME assets present in backup
5. **Training outputs**: Phase2 stage1/2/3 checkpoints recovered from backup
6. **Dataset**: StyleGAN2 generated faces (10000 images) in archive/
7. **Git**: Repository initialized on 5060, synced with GitHub
8. **Environment**: Python venv intact, CUDA rasterizer recompiled

### Known Gaps

- [待确认] Original human face dataset (non-generated) — was it restored?
- [待确认] ArcFace embeddings extraction results — were they backed up?
- [待确认] Pre-2060 experiment logs — anything missing?
- [待確認] Any unpublished experiment results lost?

## Recovery Protocol (Future Incidents)

### Immediate Actions

1. **Assess**: What hardware is affected? What data was on it?
2. **Inventory**: List all known project assets on affected machine
3. **Prioritize**: Code > Model weights > Training outputs > Datasets > Logs
4. **Restore**: From backups / GitHub / secondary machines
5. **Validate**: Smoke test the restored environment
6. **Document**: Update this file with incident details

### Prevention

- [x] Code on GitHub (3-way sync)
- [x] SSH keys backed up
- [ ] [待确认] Regular dataset backup to external drive?
- [ ] [待确认] Model checkpoint backup to cloud?
- [ ] PIL context files backed via GitHub

### Backup Checklist

Run periodically:

```bash
# On 5060:
git push origin main                      # Code is safe
# Manual: copy results/ to external drive  # Checkpoints
# Manual: verify archive.zip integrity     # Dataset
```
