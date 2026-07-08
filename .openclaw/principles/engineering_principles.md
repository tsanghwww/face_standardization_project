# Engineering Principles

## Development Philosophy

### Reproducibility First
- Every experiment must be reproducible from code + checkpoint + config
- Log exact command lines used
- Track random seeds
- Save full hyperparameter sets

### Incremental Validation
- Validate each pipeline stage independently before chaining
- Smoke test after environment changes
- Verify checkpoint loading before long training runs

### Minimal Environment
- Keep dependencies explicit in requirements files
- Avoid system-level package dependencies on 5060
- Use venv, not conda (5060 has no conda)

### Code Quality
- Python 3.12+ type hints encouraged
- Docstrings for public functions in phase2/
- No dead code — remove or comment with reason
- Use `logging` module, not `print()`, for phase2 code

### Git Hygiene
- Atomic commits with descriptive messages
- One logical change per commit
- No binary files in git (model weights, datasets are gitignored)
- Pull before push, rebase to avoid merge commits

### DECA Boundary
- DECA is a dependency, not our code
- If DECA needs patching, document the patch in docs/DECISIONS.md
- Prefer wrapping/adapter pattern over modifying DECA source
- Exceptions: the .cu and renderer.py patches already applied (documented in git history)

## Research Practices

### Experiment Tracking
- Every experiment gets an entry in docs/EXPERIMENTS.md
- Record: hypothesis, config, results, conclusion
- Tag checkpoints with experiment IDs

### Decision Recording
- Architecture decisions → docs/DECISIONS.md
- Include: context, options considered, rationale, date

### Failure Documentation
- Bug reports → docs/BUGS.md
- Include: symptoms, reproduction steps, resolution
- Recovery incidents → docs/RECOVERY.md
