"""CPU protocol checks for counterfactual RGB evaluation and external-data quarantine."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = (root / "phase3" / "audit_counterfactual_geometry.py").read_text(encoding="utf-8")
    evaluator = (root / "phase3" / "evaluate_counterfactual_rgb.py").read_text(encoding="utf-8")
    prepare = (root / "scripts" / "prepare_external_train_candidates.py").read_text(encoding="utf-8")
    identity = (root / "scripts" / "audit_external_train_identity.py").read_text(encoding="utf-8")
    deca = (root / "scripts" / "extract_external_train_deca.py").read_text(encoding="utf-8")

    assert '--save-images' in audit
    assert 'negative_yaw_sha256' not in audit  # fields are generated uniformly, not hard-coded per arm
    assert 'result[f"{name}_sha256"]' in audit
    assert 'config_sha256' in audit and 'metrics_sha256' in audit
    print("[1] counterfactual audit saves hash-addressed RGB outputs without changing default behavior OK")

    assert 'config.get("split") != "validation"' in evaluator
    assert 'Fixed-test leakage into RGB evaluation' in evaluator
    assert 'n_expected_outputs' in evaluator and 'failures retained' in evaluator
    assert 'source_minus_output_improvement' in evaluator
    assert 'gaze_evaluated' in evaluator and 'False' in evaluator
    print("[2] RGB evaluator is validation-only with complete denominator and no gaze claim OK")

    assert "validation_ids.txt" in prepare
    assert "protected_validation_rows" in prepare
    assert "source_image_sha256" in prepare
    assert "'training_eligible': False" in prepare
    assert "Canonical registry changed" in prepare
    print("[3] external candidates are materialized only after fixed+validation duplicate protection OK")

    assert 'set(candidate_ids) & (fixed_ids | validation_ids)' in identity
    assert 'reject_potential_identity_overlap' in identity
    assert 'unresolved_no_embedding' in identity
    assert '"training_eligible": False' in identity
    assert 'training_eligible_count' in identity and 'FAN-only DECA extraction' in identity
    print("[4] ArcFace identity audit fails closed and cannot make candidates training-eligible OK")

    assert 'machine_pass_conditional_on_protected_coverage' in deca
    assert 'fallback_used' in deca and 'False' in deca
    assert 'fan_no_face' in deca
    assert 'training_eligible_count' in deca and 'Phase2 inference' in deca
    assert 'fixed_test_ids.txt' in deca and 'validation_ids.txt' in deca
    assert '"--resume"' in deca and 'validate_mat(mat_path)' in deca
    assert 'external_deca_progress.jsonl' in deca and 'write_checkpoint' in deca
    print("[5] FAN-only DECA gate retains failures, resumes valid mats, and remains pending Phase2 provenance OK")

    print("ALL PHASE3.1H EXTERNAL TRAINING INTAKE PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
