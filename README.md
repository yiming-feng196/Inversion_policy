# Inversion Policy

<p align="center">
  <img src="assets/prior_flow_architecture.svg" width="780" alt="Prior Flow architecture" />
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#repository-layout">Repository Layout</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#reproducibility">Reproducibility</a> ·
  <a href="prior_policy/README.md">Method</a> ·
  <a href="real_robot/README.md">Real Robot</a>
</p>

## Overview

**Inversion Policy** learns a conditional Prior Flow over expert-inversion latents for a frozen Flow Matching policy. The Prior maps Gaussian source noise into a behavior-compatible latent; the frozen Action Flow then decodes that latent into an action chunk.

```text
observation -> frozen encoder -> condition c
Gaussian noise -> Prior Flow -> latent z -> frozen Action Flow -> action chunk
```

The current method uses **temporal thinning**: it removes redundant overlapping action windows from an expert-inversion cache while preserving temporal coverage in each demonstration. The validated deployment setting is **P8+A10**: eight Euler steps for the Prior Flow followed by ten Action Flow steps.

## Repository Layout

```text
Inversion_policy/
├── assets/                         # figures used in documentation
├── configs/                        # reproducible experiment templates
├── docs/                           # setup and reproduction instructions
├── prior_policy/
│   ├── full_data/                  # original Prior Flow over every cached window
│   ├── temporal_thinning/          # stride=8 cache derivation and training
│   └── runtime/                    # Prior sampling and closed-loop rollout
├── real_robot/                     # LeRobot v3 conversion, training, and deployment
├── theory/                         # method rationale and experiment analysis
└── legacy/                         # earlier latent-geometry and region-tracking studies
```

The active implementation is under [`prior_policy/`](prior_policy/README.md). The original full-data method is retained separately under [`prior_policy/full_data/`](prior_policy/full_data/). LeRobot v3 datasets and hardware-facing inference are documented under [`real_robot/`](real_robot/README.md). Earlier exploratory work is isolated in `legacy/` and is not part of the current training or rollout path.

## Quick Start

This project runs inside a compatible MomentVLA / RoboVerse checkout. It expects a frozen Action Flow checkpoint and an offline expert-inversion cache; neither is distributed in this repository.

```bash
# 1. Install the Python dependencies in the MomentVLA environment.
pip install -r requirements.txt

# 2. Build a stride=8 subset from an existing full inversion cache.
python prior_policy/temporal_thinning/build_stride8_cache.py \
  --source-cache /path/to/full_inversion_cache \
  --output-cache /path/to/stride8_inversion_cache \
  --stride 8

# 3. Train a Prior with a fixed update budget.
python prior_policy/temporal_thinning/train_stride8_prior.py \
  --repo /path/to/MomentVLA-main \
  --cache /path/to/stride8_inversion_cache \
  --output-dir /path/to/output/prior_stride8 \
  --action-flow-checkpoint /path/to/action_flow.ckpt \
  --epochs 150 --max-train-steps 250 --train-all
```

See [setup](docs/SETUP.md), [reproducibility](docs/REPRODUCIBILITY.md), and the [method guide](prior_policy/README.md) for full commands and evaluation settings.

## Reproducibility

The experiment templates in `configs/` document the unified temporal-thinning protocol. They keep the Action Flow frozen and use `batch_size=32`, `150 x 250 = 37,500` optimizer updates, and all cached rows for Prior training.

| Task | Training cache | Updates | Training rows | Inference |
|---|---:|---:|---|---:|
| PickCube | stride=8 | 37,500 | all cached rows | P8+A10 |
| StackCube | stride=8 | 37,500 | all cached rows | P8+A10 |
| CloseBox | stride=8 | 37,500 | all cached rows | P8+A10 |

Earlier 7,500-step runs are pilot ablations, not part of this standardized comparison. Full details, comparison boundaries, and rollout variation are recorded in [`theory/experimental_observations.md`](theory/experimental_observations.md).

## Citation

If you use this repository, please cite the associated work once available. Until then, link directly to this repository and identify the commit hash used for the experiment.

## License and Data

This repository contains source code and documentation only. Checkpoints, demonstrations, cached inversion latents, videos, and rollout outputs are intentionally excluded. Use the licensing terms of the underlying MomentVLA / RoboVerse project and task assets.
