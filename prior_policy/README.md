# Prior Policy

This directory contains the active Inversion Policy implementation.

## Components

| Directory | Purpose |
|---|---|
| `full_data/` | Original Prior Flow trained on every row of the inversion cache. Kept as the full-data reference. |
| `temporal_thinning/` | Current method: derives a stride=8 cache and trains under a fixed optimizer-update budget. |
| `runtime/` | Frozen Action Flow interface, Prior Flow sampling, and paired closed-loop rollout. |

## Method

For a frozen Action Flow `F_theta` and condition `c`, offline inversion produces an expert latent `z*`. The Prior Flow `G_phi` learns a conditional vector field from Gaussian noise `epsilon` to `z*`:

```text
z_t = (1 - t) epsilon + t z*
u   = z* - epsilon
L   = MSE(v_phi(z_t, t, c), u)
```

At rollout time:

```text
Gaussian epsilon -> Prior Flow G_phi -> z_prior -> frozen Action Flow F_theta -> action
```

The Action Flow, visual encoder, normalizer, and cached inversion targets are frozen during Prior training.

## Full-data reference

[`full_data/train_full_data_prior.py`](full_data/train_full_data_prior.py) consumes every cache row. It is preserved as the reference implementation for comparisons with temporal thinning.

## Temporal thinning

[`temporal_thinning/build_stride8_cache.py`](temporal_thinning/build_stride8_cache.py) selects cache rows inside each `(split, episode)` group:

```text
sorted windows: 0, 1, 2, ..., final
selected:       0, 8, 16, ..., final
```

The final window is always retained, which preserves terminal task behavior. [`temporal_thinning/train_stride8_prior.py`](temporal_thinning/train_stride8_prior.py) cycles shuffled passes over the thinned cache to execute the requested optimizer-step budget exactly. The unified configuration uses `batch_size=32`, 150 epochs, 250 steps per epoch, and `--train-all` for 37,500 optimizer updates.

## Runtime

Use [`runtime/prior_flow_rollout.py`](runtime/prior_flow_rollout.py) for paired FM and Prior rollouts. The validated deployment configuration is `--prior-steps 8 --action-steps 10`.

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for complete commands.
