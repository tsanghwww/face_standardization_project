# Phase3.1g High-Resolution Geometry-Delta Protocol

## Decision

Phase3.1f is stopped before validation. Its median projected response was only
0.084 degrees for a roughly 20-degree same-identity target change, while source
reconstruction geometry degraded. Phase3.1g changes the injection mechanism;
it is not a longer continuation of the Phase3.1f run.

## Root-Cause Changes

1. The source reconstruction checkpoint is frozen in full. Source inference
   uses the original adapter path, so target-control training cannot degrade it.
2. The edit branch consumes `target_condition - source_condition` at 256x256.
   It encodes at 256, 128, and 64 before producing residuals for the UNet's
   32, 16, 8, and 4 spatial scales. The previous adapter first resized the
   complete condition to 32x32.
3. The edit residual is added to the frozen source residual pyramid. A hard
   binary gate makes `target == source` an exact zero edit before and after
   training.
4. Negative-yaw and positive-yaw arms share identity, source latent, noise,
   and timestep. In addition to target error and target-vs-source ranking, a
   minimum signed pair-separation loss requires the output change projected on
   the target rotation axis to reach 10% of the target pair separation. An
   opposite-direction change increases this loss instead of satisfying it.
5. Timesteps remain restricted to 100-400. Validation and fixed test remain
   sealed until the train-only mechanism gate passes.

## Objective

For counterfactual arms `a in {negative_yaw, positive_yaw}`:

```text
L = 0.003 * mean_a L_geometry(a)
  + 1.0   * mean_a max(0, margin + D(output_a, target_a)
                                 - D(source_output, target_a))
  + 1.0   * max(0, r * Delta_target - project(Delta_output, axis_target))
            / max(r * Delta_target, epsilon)
```

The initial transfer ratio is `r = 0.10`. This is a bounded mechanism test,
not a final production loss or a claim of gaze disentanglement.

## Required Checks

- CPU protocol test passes.
- Warm-start model, split, input, counterfactual, and code hashes are recorded.
- Source no-op epsilon is bitwise equal to the frozen reconstruction adapter.
- Both counterfactual arms yield finite, positive residual-output gradients.
- Preflight peak allocated GPU memory is below 7.2 GiB.
- Every selected optimization ID is included in the audit denominator.
- No validation or fixed-test ID is used for training or threshold selection.

## Train-Only Progression

1. Run a 2-step GPU smoke on two train identities.
2. If all checks pass, run a 64-step bounded overfit on eight train identities.
3. Audit those eight identities at timesteps 100, 250, and 400.
4. Continue to the frozen validation split only when the predeclared
   counterfactual audit gate passes: complete denominator, at least 75% of rows
   with correct direction and own-target ordering, and median projected pose
   change of at least 0.5 degrees.

The validation audit uses `--split validation`, verifies that the checkpoint's
recorded training inputs do not overlap the validation IDs, and keeps the fixed
test registry sealed. It applies the already-declared gate without changing its
thresholds.

On a host with an already compiled DECA standard rasterizer but no active MSVC
toolchain, the condition builder may use `--prebuilt-rasterizer`. The binary
must exist as `standard_rasterize_cuda.pyd`; its SHA256 is recorded in the
summary. This bypasses JIT build discovery without changing DECA rendering.

Passing this gate only demonstrates that the new branch can learn geometry
control on optimization identities. It does not establish generalization,
identity preservation, standardization quality, or gaze disentanglement.
