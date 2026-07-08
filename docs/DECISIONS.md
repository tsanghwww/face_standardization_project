# Technical Decisions

## DEC-001: Use frozen DECA with learned condition generator

- **Date**: [待确认] (pre-PIL)
- **Decision**: Instead of fine-tuning DECA, use a separate Phase2 condition generator that learns to map DECA outputs to standardized parameters
- **Rationale**: Preserves DECA pretraining, allows independent iteration on standardization logic, avoids catastrophic forgetting
- **Alternatives considered**: Fine-tuning DECA end-to-end (rejected: risk of degrading reconstruction quality)

## DEC-002: XGBoost quality scorer

- **Date**: [待确认] (pre-PIL)
- **Decision**: Train an XGBoost model to predict output quality, use scores to weight training loss
- **Rationale**: Provides quality-aware training signal without requiring manual annotation
- **Alternatives considered**: Manual quality scoring (rejected: not scalable), learned quality network (rejected: overkill for initial version)

## DEC-003: PyTorch 2.6+ compatibility patches

- **Date**: 2026-07-08
- **Decision**: Apply minimal patches to DECA CUDA rasterizer (.data() → .data_ptr(), .type() → .scalar_type()) and add weights_only=False to torch.load
- **Rationale**: Necessary for PyTorch 2.6+ compatibility; patches are minimal and isolated
- **Impact**: Modified `standard_rasterize_cuda_kernel.cu`, `renderer.py`, `phase2/infer_standardize_params.py`

## DEC-004: PIL as overlay, not replacement

- **Date**: 2026-07-08
- **Decision**: Implement Project Intelligence Layer as .openclaw/ + docs/ overlay, zero modifications to existing project structure
- **Rationale**: Non-invasive, keeps project structure clean, easy to remove if needed
- **Alternatives considered**: Restructuring project for AI-friendliness (rejected: unnecessary risk)

## DEC-005: Markdown-only context files

- **Date**: 2026-07-08
- **Decision**: All PIL context files use plain Markdown, no JSON/YAML/DB
- **Rationale**: Git-friendly (readable diffs), agent-friendly (no parsing overhead), human-readable
- **Alternatives considered**: Structured JSON (rejected: harder to read diffs), SQLite (rejected: overengineered)
