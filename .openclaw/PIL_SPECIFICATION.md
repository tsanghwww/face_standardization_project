# PIL Specification — face_standardization_project

## What PIL Is

PIL (Project Intelligence Layer) is an AI-maintainable context overlay on top of the existing project. It does NOT replace or restructure the project. It adds a set of Markdown files that AI agents read at session start and update at session end, enabling long-term project memory across sessions, agents, and context resets.

## File Layout

```
.openclaw/                      Agent runtime layer (rules, workflows, skills)
  PIL_SPECIFICATION.md          This file — PIL self-description
  principles/                   Engineering and research principles
  workflows/                    Standard operating procedures
  skills/                       Agent-triggerable skill definitions
  rules/                        Hard constraints and style guides

docs/                           Project long-term memory
  CONTEXT.md                    Project overview and architecture summary
  STATUS.md                     Current phase, progress, blockers
  TODO.md                       Task list with priorities
  NEXT.md                       Immediate next actions (resume point)
  LOG.md                        Chronological operation log (append-only)
  DECISIONS.md                  Important technical decisions
  EXPERIMENTS.md                Experiment records and results
  BUGS.md                       Bug lifecycle tracking
  RECOVERY.md                   Disaster recovery record and lessons
  PIPELINE.md                   Data/training pipeline documentation
  MODEL.md                      Model architecture and checkpoint index
  DATASET.md                    Dataset inventory and locations
  ENVIRONMENT.md                Development environment configuration
  PAPERS.md                     Related papers and literature
```

## Context Loading Protocol

Every agent session MUST begin with context loading:

1. Read PIL_SPECIFICATION.md (this file)
2. Read docs/CONTEXT.md
3. Read docs/STATUS.md
4. Read docs/TODO.md
5. Read docs/NEXT.md
6. Read last 50 lines of docs/LOG.md
7. Read docs/DECISIONS.md
8. Read docs/RECOVERY.md

Generate summary: project name, current phase, completed items, pending items, current task, risks, code modification boundaries.

## Context Saving Protocol

At end of each significant work session, check and update:

- STATUS.md    — if project phase/progress changed
- TODO.md      — if tasks completed or added
- NEXT.md      — update next steps
- LOG.md       — append session summary (timestamped)
- DECISIONS.md — if any technical decision was made
- EXPERIMENTS.md — if experiments were run
- BUGS.md      — if bugs discovered or resolved
- RECOVERY.md  — if any recovery actions were taken

## Modification Boundaries

- DECA/ — DO NOT MODIFY (third-party code)
- phase2/ — core research code, modify with care
- tools/ — utility scripts, modifiable
- docs/ — PIL memory files, freely modifiable
- .openclaw/ — agent layer, freely modifiable

## Version

PIL v1.0 — initialized 2026-07-08
