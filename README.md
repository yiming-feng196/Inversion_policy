# Inversion Policy

Research code for a learned Prior Flow that maps Gaussian source noise to expert-inversion action latents before a frozen Action Flow produces executable actions.

## Layout

- `prior_policy/full_data_prior/`: the original, full-cache Prior Flow trainer.
- `prior_policy/temporal_thinning/`: the current stride=8 implementation and cache tools.
- `prior_policy/runtime/`: frozen Action Flow interface, prior inference, and closed-loop rollout entry point.
- `theory/`: method rationale and experimental analysis.

## Dependency

These modules run inside the MomentVLA / Roboverse codebase. They intentionally retain imports such as `roboverse_learn.*`; place this repository alongside that codebase or add it to `PYTHONPATH`.

Model weights, datasets, cached inversions, videos, and rollout outputs are intentionally excluded.
