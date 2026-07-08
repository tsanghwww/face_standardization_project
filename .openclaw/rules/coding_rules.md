# Coding Rules

## Python

### Style
- Follow PEP 8
- 4-space indentation
- 100-character line limit (soft)
- Use `black` formatting when available

### Imports
- Standard library first, then third-party, then local
- Avoid `import *`
- Use absolute imports in phase2/

### Type Hints
- Encourage for public function signatures in phase2/
- Not required for internal helpers or DECA code

### Error Handling
- Explicit try/except, never bare `except:`
- Log exceptions with tracebacks
- Graceful degradation for batch processing (skip bad samples, report counts)

### Paths
- Use `pathlib.Path`, not `os.path`
- Make paths configurable via CLI args, not hardcoded
- Use relative paths from project root where possible

### GPU Code
- Always check `torch.cuda.is_available()` before CUDA operations
- Provide CPU fallback in inference scripts
- Clear CUDA cache between large operations if needed

## Shell Scripts (.bat / .ps1)

- Keep scripts idempotent (safe to re-run)
- Use absolute paths for project directory
- Activate venv explicitly at script start

## Configuration

- Use argparse for CLI scripts
- Store hyperparameters in the training script or a JSON config
- Avoid hardcoding paths

## File Naming

- Python files: `snake_case.py`
- Batch files: `snake_case.bat`
- Experiment outputs: `{experiment_name}_{date}/`
