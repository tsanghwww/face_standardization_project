# Safety Rules

## Hard Constraints

### DECA/ — Third-Party Code (Read-Only)
- NEVER modify files under DECA/
- NEVER move or rename DECA/ or its subdirectories
- NEVER refactor DECA/ code
- If DECA compatibility issue arises, document in BUGS.md and flag as [DECA]

### File Safety
- NEVER delete files without explicit user confirmation
- NEVER move project directories
- NEVER overwrite historical documents (*.docx, *.tex, *.pdf)
- Before destructive operations, list affected files and ask

### Code Safety
- Before modifying phase2/ or tools/, read the full target file
- Preserve existing function signatures unless explicitly asked
- Add comments for non-obvious changes

### Model & Data Safety
- NEVER delete model weights or checkpoints
- NEVER delete dataset files
- NEVER modify DECA/data/ contents

## SSH / Remote Operations
- Prefer dry-run before destructive remote commands
- Confirm 5060 connection before batch operations
- Log all remote operations in LOG.md

## Git Safety
- Always `git pull --rebase` before pushing
- Never force-push to main
- Review `git status` before committing
- Exclude large binary files (model weights, datasets) from git — they are already in .gitignore

## Environment
- Mac: primary control node
- 5060 (win-lenovo): training/inference node via SSH
- GitHub: code sync hub
- Do NOT install system-level packages on 5060 without asking
