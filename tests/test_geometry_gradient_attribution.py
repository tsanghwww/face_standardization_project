"""Verify sequential ranking decomposition against a joint autograd graph."""
import torch
from phase3.audit_geometry_gradients import attribution, cosine, joint_directional_audit


def main():
    p = torch.tensor([.3, -.2], requires_grad=True)
    before = p.detach().clone()
    for offset in (0., -100.):
        dt = (p * torch.tensor([2., 3.])).square().sum() + offset
        ds = (p * torch.tensor([1., 4.])).square().sum()
        gt = torch.autograd.grad(dt, p, retain_graph=True)[0]
        gs = torch.autograd.grad(ds, p, retain_graph=True)[0]
        ranking = torch.clamp(.05 + dt - ds, min=0)
        direct = torch.autograd.grad(ranking, p)[0]
        decomposed = (gt-gs) * float(.05+dt.detach()-ds.detach() > 0)
        assert torch.allclose(direct, decomposed)
    assert p.grad is None and torch.equal(p, before)
    v = {'src': torch.tensor([1., 0.]), 'geometry': torch.tensor([0., 2.]),
         'ranking': torch.tensor([1., -1.]), 'd_target': torch.tensor([2., 1.]),
         'd_source': torch.tensor([1., 2.])}
    result = attribution(v, {'src': 1., 'geometry': 1., 'ranking': .1}, {0: torch.tensor([True, False])})
    assert abs(result['weighted_norms']['ranking'] - .1*2**.5) < 1e-6
    assert cosine(torch.zeros(2), torch.ones(2)) is None
    assert result['per_scale']['0']['geometry'] == 0
    full = {
        'src': torch.tensor([1., 0., 0., 0.]),
        'geometry': torch.tensor([0., 1., 0., 0.]),
        'source_geometry': torch.tensor([0., 0., 1., 0.]),
        'ranking': torch.tensor([0., 1., 0., 0.]),
        'd_target': torch.tensor([0., 1., 0., 0.]),
        'd_source': torch.zeros(4),
        'd_source_self': torch.tensor([0., 0., 1., 0.]),
    }
    joint = joint_directional_audit(full, geometry_weight=.003, ranking_weight=.1)
    assert joint['target_distance_dd'] < 0
    assert joint['target_source_gap_dd'] < 0
    assert joint['source_mse_dd'] < 0
    assert joint['source_self_distance_dd'] < 0
    assert joint['all_four_improve']
    print('Gradient decomposition, inactive hinge, no mutation, and weighted attribution passed')


if __name__ == '__main__':
    main()
