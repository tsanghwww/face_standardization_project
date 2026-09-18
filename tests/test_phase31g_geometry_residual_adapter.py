"""CPU protocol tests for the Phase3.1g high-resolution delta adapter."""

from pathlib import Path

import torch

from phase3.geometry_residual_adapter import (
    HighResolutionGeometryEditAdapter,
    minimum_pair_separation_loss,
)
from phase3.differentiable_geometry import relative_rotation_vector_deg


def main() -> None:
    adapter = HighResolutionGeometryEditAdapter(output_channels=(8, 16, 24, 32))
    source = torch.rand(2, 6, 256, 256)
    target = source.clone()
    outputs = adapter(source, target, (32, 32))
    assert [tuple(value.shape) for value in outputs] == [
        (2, 8, 32, 32), (2, 16, 16, 16), (2, 24, 8, 8), (2, 32, 4, 4),
    ]
    assert all(torch.count_nonzero(value) == 0 for value in outputs)

    # A zero geometry delta must remain an exact no-op after the edit branch has
    # learned nonzero weights and biases.
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.uniform_(-0.1, 0.1)
    no_op = adapter(source, source, (32, 32))
    assert all(torch.count_nonzero(value) == 0 for value in no_op)

    changed = source.clone()
    changed[:, 1] += 0.1
    edited = adapter(source, changed, (32, 32))
    assert all(torch.isfinite(value).all() for value in edited)
    assert any(torch.count_nonzero(value) > 0 for value in edited)
    sum(value.square().mean() for value in edited).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in adapter.parameters())

    output_separation = torch.tensor(1.0, requires_grad=True)
    loss, required = minimum_pair_separation_loss(output_separation, torch.tensor(20.0), 0.1)
    assert torch.allclose(required, torch.tensor(2.0)) and torch.allclose(loss, torch.tensor(0.5))
    loss.backward()
    assert torch.allclose(output_separation.grad, torch.tensor(-0.5))
    satisfied, _ = minimum_pair_separation_loss(torch.tensor(2.5), torch.tensor(20.0), 0.1)
    assert satisfied.item() == 0.0
    reversed_loss, _ = minimum_pair_separation_loss(torch.tensor(-1.0), torch.tensor(20.0), 0.1)
    assert torch.allclose(reversed_loss, torch.tensor(1.5))

    negative = torch.tensor([[0.0, -torch.pi / 18, 0.0]], requires_grad=True)
    positive = torch.tensor([[0.0, torch.pi / 18, 0.0]], requires_grad=True)
    signed_delta = relative_rotation_vector_deg(negative, positive)
    assert torch.allclose(signed_delta, torch.tensor([[0.0, 20.0, 0.0]]), atol=1e-4)
    signed_delta.sum().backward()
    assert torch.isfinite(negative.grad).all() and torch.isfinite(positive.grad).all()
    zero = torch.zeros(1, 3, requires_grad=True)
    identical = relative_rotation_vector_deg(zero, zero)
    assert torch.allclose(identical, torch.zeros_like(identical), atol=1e-7)
    identical.sum().backward()
    assert torch.isfinite(zero.grad).all()

    root = Path(__file__).parents[1]
    training_source = (root / "phase3" / "train_geometry_residual_control.py").read_text(encoding="utf-8")
    audit_source = (root / "phase3" / "audit_counterfactual_geometry.py").read_text(encoding="utf-8")
    assert 'GeometryAuditDataset(args.manifest, args.split_dir, args.ids_file, "train")' in training_source
    assert "100 <= args.geometry_timestep_low <= args.geometry_timestep_high <= 400" in training_source
    assert '"source_no_op_exact": True' in training_source
    assert "expected - set(by_id)" in training_source
    assert 'saved.get("architecture", "face_control_adapter_v1")' in audit_source
    assert "set(ids) - set(counterfactuals)" in audit_source
    print("Phase3.1g high-resolution delta adapter protocol passed")


if __name__ == "__main__":
    main()
