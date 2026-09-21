# Contributing

## Scope

Keep active method changes inside `prior_policy/`, method analysis inside `theory/`, and earlier exploratory work inside `legacy/`. Do not add datasets, checkpoints, inversion caches, rollout videos, or simulator outputs to Git.

## Pull requests

- State the task, cache selection rule, update budget, and P/A inference steps.
- Preserve Action Flow checkpoint compatibility checks.
- Report seed-level rollout counts and paired conditions for empirical claims.
- Update `docs/REPRODUCIBILITY.md` and the relevant `theory/` note when changing an experiment protocol.

## Code style

Use Python 3.8+ compatible syntax, explicit CLI arguments, deterministic seed initialization, and atomic artifact writes. Retain the existing `roboverse_learn.*` integration boundary instead of duplicating simulator internals.
