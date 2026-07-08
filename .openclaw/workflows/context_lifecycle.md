# Context Lifecycle Workflow

## Session Start (Context Loader)

Every AI agent session in this project MUST start with context loading.

### Load Order

1. `.openclaw/PIL_SPECIFICATION.md` — understand PIL mechanism and boundaries
2. `docs/CONTEXT.md` — project overview
3. `docs/STATUS.md` — current phase and progress
4. `docs/TODO.md` — task list
5. `docs/NEXT.md` — immediate next actions
6. `docs/LOG.md` — last 50 lines for recent activity
7. `docs/DECISIONS.md` — technical constraints
8. `docs/RECOVERY.md` — known risks and lessons

### After Loading

Generate and present:

- Project summary (1 paragraph)
- Current phase and progress
- Completed items
- Pending items
- Current task
- Active risks and constraints
- Modification boundaries (DECA = read-only)

### Pre-Work Checks

- Is 5060 reachable? (if remote work needed)
- Is GitHub in sync?
- Are model checkpoints accessible?
- Any `[待确认]` items that block progress?

## Session End (Context Saver)

### Update Checklist

| File | When to Update |
|---|---|
| STATUS.md | Phase changed, progress milestone hit, blocker appeared/resolved |
| TODO.md | Tasks completed, new tasks identified |
| NEXT.md | ALWAYS — what should the next session do first? |
| LOG.md | ALWAYS — append timestamped session summary |
| DECISIONS.md | Technical decision made (architecture, approach, tooling) |
| EXPERIMENTS.md | Experiment completed (even if negative result) |
| BUGS.md | Bug discovered or resolved |
| RECOVERY.md | Any recovery actions taken |

### LOG.md Format

```markdown
## YYYY-MM-DD HH:MM — Session Summary

- **Agent**: DeepSeek / GPT
- **Tasks performed**:
  - Item 1
  - Item 2
- **Files changed**: file1.py, file2.md
- **Key outcomes**: description
- **Risks noted**: if any
```

### Git Commit

After updating docs/, commit PIL changes:

```bash
git add docs/ .openclaw/
git commit -m "PIL: update context after [session summary]"
git push origin main
```

## Cross-Agent Protocol

When switching between DeepSeek (this agent) and GPT:

- DeepSeek updates all docs/ files at session end
- GPT reads docs/ files at session start
- Both agents follow the same load/save protocol
- File format is plain Markdown — no agent-specific syntax
