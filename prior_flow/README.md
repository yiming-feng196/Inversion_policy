# Expert-Inversion Prior Flow

This directory contains the core Prior Flow implementation used with the
frozen MomentVLA Action Flow policy.

The inference composition is:

```text
epsilon ~ N(0, I)
  -> Prior Flow G_phi(epsilon, c)
  -> frozen Action Flow F_theta(z_prior, c)
  -> action
```

Only `G_phi` is trained. The Action Flow, visual encoder, normalizer, and
their parameters remain frozen.

## Files

- `expert_inversion_prior_flow.py`: train `G_phi` from cached detached
  `(condition, expert-inversion latent)` pairs.
- `prior_flow_inference.py`: Prior sampling and frozen Action Flow composition.
- `prior_flow_rollout.py`: IsaacSim rollout runner for FM and Prior methods,
  including deterministic source seeds and paired traces.

## Integration

The scripts are designed to run inside a MomentVLA checkout. Copy them to:

```text
roboverse_learn/il/policies/fm/latent_prediction/
```

They require an existing compatible frozen Action Flow checkpoint and an
inversion cache. Checkpoints, caches, datasets, videos, and experiment outputs
are intentionally not included in this repository.

The Prior checkpoint records the SHA-256 of the Action Flow checkpoint used to
create its inversion targets. Rollout rejects mismatched checkpoints by
default; use `--allow-action-checkpoint-mismatch` only for an explicitly
documented cross-decoder experiment.

## Training objective

For a cached expert endpoint `z_star` and frozen condition `c`, sample
`epsilon ~ N(0,I)` and `t ~ U(0,1)`:

```text
z_t = (1 - t) * epsilon + t * z_star
u   = z_star - epsilon
loss = MSE(v_phi(z_t, t, c), u)
```

The default network matches the Action Flow ConditionalUnet1D architecture.
