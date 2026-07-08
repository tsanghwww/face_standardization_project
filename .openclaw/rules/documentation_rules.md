# Documentation Rules

## Project Documentation (docs/)

### File Purposes

| File | Content | Update Trigger |
|---|---|---|
| CONTEXT.md | Project overview, architecture, current state | Major project changes |
| STATUS.md | Current phase, progress %, blockers | Phase transitions, blockers change |
| TODO.md | Prioritized task list | Tasks completed or added |
| NEXT.md | Immediate next 1-3 actions | After each session |
| LOG.md | Timestamped operation records | Each significant action |
| DECISIONS.md | Technical decisions with rationale | Each architecture/approach decision |
| EXPERIMENTS.md | Experiment records with results | Each experiment run |
| BUGS.md | Bug reports with lifecycle | Bug discovery, progress, resolution |
| RECOVERY.md | Recovery incidents and lessons | Any data loss or recovery event |
| PIPELINE.md | Data/training pipeline steps | Pipeline changes |
| MODEL.md | Model architecture, checkpoint index | New checkpoints, architecture changes |
| DATASET.md | Dataset inventory, locations, formats | Dataset additions or changes |
| ENVIRONMENT.md | Hardware, software, config details | Environment changes |

### Formatting Rules

- All files in Markdown
- Use `##` headers for sections
- Timestamps in ISO 8601: `YYYY-MM-DD HH:MM`
- Use `[待确认]` for uncertain information
- Use checkboxes `- [ ]` / `- [x]` in TODO.md
- LOG.md entries: `## YYYY-MM-DD HH:MM — Summary Title` followed by bullet points

### Ownership

- docs/ files are maintained by AI agents
- Human can edit directly, but agents will re-read on next session
- Critical decisions should be confirmed by human before recording in DECISIONS.md

## Code Documentation

- README.md: kept current with project overview
- phase2/README.md: kept current with phase2 usage
- DECA/RUNNING_MODERN.md: reference for DECA setup (third-party, read-only)
- Docstrings: encouraged for public functions in phase2/
