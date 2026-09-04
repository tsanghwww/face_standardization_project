"""Numerical protocol checks for source-latent DDIM sampling."""

import torch
from diffusers import DDIMScheduler

from phase3.sample_latent_img2img import img2img_timesteps, sample_latent, tensor_hash, load_adapter_exact
from phase3.evaluate_latent_img2img import paired_summary, stats


def test_sampling():
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                              beta_schedule='scaled_linear', steps_offset=1,
                              set_alpha_to_one=False, clip_sample=False)
    source = torch.randn((1, 4, 8, 8), generator=torch.Generator().manual_seed(1))
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(2))
    calls = []

    def predict(value, timestep):
        calls.append(int(timestep))
        return noise

    zero, initial = sample_latent(source, noise, scheduler, 20, 0, predict)
    assert torch.equal(source, zero) and initial is None and not calls
    for strength, count, first in [(0.25, 5, 201), (0.5, 10, 451), (0.75, 15, 701), (1, 20, 951)]:
        ts = img2img_timesteps(scheduler, 20, strength)
        assert len(ts) == count and int(ts[0]) == first
        calls.clear()
        actual, initial = sample_latent(source, noise, scheduler, 20, strength, predict)
        expected_initial = scheduler.add_noise(source, noise, ts[:1])
        assert initial == tensor_hash(expected_initial)
        assert calls == ts.tolist()
        # An oracle epsilon predicts the same forward trajectory down to scheduler final alpha.
        a = scheduler.final_alpha_cumprod
        expected = a.sqrt() * source + (1-a).sqrt() * noise
        assert torch.allclose(actual, expected, atol=2e-5), (actual-expected).abs().max()
        repeated, repeated_hash = sample_latent(source, noise, scheduler, 20, strength, predict)
        assert torch.equal(actual, repeated) and initial == repeated_hash
    assert not torch.equal(expected_initial, noise), 'strength=1 must retain source contribution'
    for strength in (-0.1, 1.1, float('nan'), float('inf'), 0.01):
        try:
            img2img_timesteps(scheduler, 20, strength)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid strength accepted: {strength}')
    for steps in (0, 1001):
        try:
            img2img_timesteps(scheduler, steps, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid step budget accepted')
    print('DDIM schedule, zero-strength bypass, source/noise initialization, oracle trajectory, reproducibility: OK')


def test_audit_and_checkpoint():
    rows = [{'image_id': 'a', 'variant': 'frozen', 'cosine': 0.1},
            {'image_id': 'a', 'variant': 'trained', 'cosine': 0.3},
            {'image_id': 'b', 'variant': 'frozen', 'cosine': 0.9},
            {'image_id': 'b', 'variant': 'trained', 'cosine': None}]
    summary = paired_summary(rows, ['a', 'b', 'c'], 'cosine')
    assert summary['n_total'] == 3 and summary['n_paired'] == 1
    assert abs(summary['trained_minus_frozen']['mean'] - 0.2) < 1e-6
    assert stats([None])['mean'] is None and stats([None])['count'] == 0

    class Model:
        def adapter_state(self):
            return {'face.weight': torch.ones(2)}

        def load_state_dict(self, state, strict):
            self.loaded = state

    model = Model()
    load_adapter_exact(model, model.adapter_state())
    for state in ({}, {'unet.weight': torch.ones(2)}, {'face.weight': torch.ones(3)},
                  {'face.weight': torch.tensor([float('nan'), 1])}):
        try:
            load_adapter_exact(model, state)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid adapter checkpoint accepted')
    print('Paired denominators, null metrics, strict adapter key/shape/finite validation: OK')


if __name__ == '__main__':
    test_sampling()
    test_audit_and_checkpoint()