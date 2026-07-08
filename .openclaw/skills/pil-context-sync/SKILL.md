---
name: pil-context-sync
description: "Initialize PIL structure, load project context, save session state, and perform close-session sync for face_standardization_project."
allowed-tools: read,write,edit,exec,memory_search,memory_get
---

# PIL Context Sync

Manages the Project Intelligence Layer lifecycle for face_standardization_project.

## Workflows

### init

Create PIL directory structure and seed all context files from templates.

Do NOT overwrite existing files. Skip files that already exist.

Steps:
1. Create directories: `.openclaw/{principles,workflows,skills,rules}` and `docs/`
2. Create each file only if missing
3. After creation, commit to git

### load

Read all context files and generate a project summary.

Load order:
1. `.openclaw/PIL_SPECIFICATION.md`
2. `docs/CONTEXT.md`
3. `docs/STATUS.md`
4. `docs/TODO.md`
5. `docs/NEXT.md`
6. `docs/LOG.md` (last 50 lines)
7. `docs/DECISIONS.md`
8. `docs/RECOVERY.md`

Output these sections:
- **Project**: name, domain, current phase
- **Progress**: completed, in-progress, pending
- **Current Task**: highest priority TODO
- **Next Actions**: from NEXT.md
- **Risks**: from STATUS.md and RECOVERY.md
- **Boundaries**: DECA = read-only, code modification rules

### save

Check for state changes since last save and update relevant docs/ files.

Update logic:
- STATUS.md → if project phase, progress, or blockers changed
- TODO.md → if tasks completed or added (mark completed with `[x]`)
- NEXT.md → ALWAYS update with immediate next actions
- LOG.md → ALWAYS append timestamped session summary
- DECISIONS.md → if any architecture/approach decision was made
- EXPERIMENTS.md → if experiments were run
- BUGS.md → if bugs discovered, progressed, or resolved
- RECOVERY.md → if any recovery actions taken

LOG.md entry format:
```
## YYYY-MM-DD HH:MM — Summary Title
- **Agent**: DeepSeek / GPT
- **Tasks performed**: ...
- **Files changed**: ...
- **Key outcomes**: ...
```

### close-session

Full end-of-session procedure:
1. Run `save` (update all context files)
2. Stage PIL changes: `git add docs/ .openclaw/`
3. Commit: `git commit -m "PIL: session sync YYYY-MM-DD"`
4. Push to GitHub: `git push origin main`
5. If 5060 is reachable, sync: `ssh win-lenovo "cd /d D:\face_standardization_project && git pull origin main"`

## Environment Awareness

- Mac: `~/Documents/face_standardization_project/`
- 5060: `D:\face_standardization_project\`
- GitHub: `tsanghwww/face_standardization_project`

## Modification Boundaries

- `DECA/` — DO NOT TOUCH (third-party, read-only)
- `phase2/` — core code, modify with care
- `tools/` — utility scripts
- `docs/` — PIL context (freely modifiable)
- `.openclaw/` — agent layer (freely modifiable)

## Safety

- NEVER delete files without user confirmation
- NEVER modify DECA/ contents
- NEVER overwrite historical documents
- Mark uncertain information as `[待确认]`
- Prefer appending to files over replacing content (LOG, DECISIONS, EXPERIMENTS, BUGS)
