# Inversion Policy: Frozen Flow Latent Experiments

This repository contains the reproducible code for studying expert inversion
latents under a frozen Flow Matching policy. The experiments test latent
relevance, global distribution structure, local behavior-specific regions, and
directional behavioral geometry.

The code does not include the MomentVLA/RoboVerse checkout, datasets,
checkpoints, latent caches, or generated results. Those assets must be supplied
through the command-line arguments.

## Core scripts

- `flow_latent_predictor_common.py`: frozen Flow loading, differentiable forward
  sampling, cache loading, normalization, and integrity helpers.
- `prepare_inverted_latent_dataset.py`: reverse-integrates expert action chunks
  to create a resumable `z_star` train/validation cache.
- `conditional_latent_relevance_test.py`: compares expert inversion with one
  Gaussian sample and Random Oracle Best@100 under fixed conditions.
- `native_cycle_vs_expert_distribution_audit.py`: audits native, cycle, and
  expert latent distributions, covariance spectra, effective rank, and
  whitening statistics.
- `local_latent_region_test.py`: compares local sampling around `z_star` with
  global Gaussian sampling.
- `directional_latent_geometry_test.py`: compares equal-length radial outward,
  radial inward, and tangential perturbations.
- `latent_geometry_figures.py`: computes the exact local Flow Jacobian, tangent
  sensitivity spectrum, and radial/tangential behavior grids. Its numerical
  outputs can be retained independently of its optional plotting code.

All experiments use a frozen Flow checkpoint and 200-step forward/reverse Flow
integration. The Flow parameters are never updated.

## Environment

Install the Python dependencies listed in `requirements.txt`. The original
MomentVLA/RoboVerse source tree is also required because the scripts instantiate
its policy and Flow matcher.

## Typical workflow

First build the inversion cache:

```bash
python prepare_inverted_latent_dataset.py \
  --repo /path/to/MomentVLA-main \
  --checkpoint /path/to/30.ckpt \
  --zarr /path/to/dataset.zarr \
  --cache /path/to/inversion_cache
```

Then run the validation scripts with the same `--repo`, `--checkpoint`, and
`--cache` arguments:

```bash
python conditional_latent_relevance_test.py \
  --repo /path/to/MomentVLA-main \
  --checkpoint /path/to/30.ckpt \
  --cache /path/to/inversion_cache \
  --output-dir /path/to/relevance_output
```

The other analysis scripts expose the same common arguments. Keep checkpoints,
datasets, caches, and generated outputs outside Git.

## Current validated findings

The validated experiments show that:

1. Expert inversion is strongly behaviorally relevant under fixed conditions.
2. Expert variance is concentrated into fewer dominant latent modes.
3. Whitening removes second-order covariance structure but leaves higher-order
   non-Gaussian radial structure.
4. Around an inverted latent, finite-radius behavior forms an asymmetric and
   directionally anisotropic local region.
