# EIP: Expert-Inversion Priors for Flow-Matching Robot Policies

<p align="center">
  <img src="assets/prior_flow_architecture.svg" width="820" alt="Expert-Inversion Prior Flow overview" />
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#reproduction">Reproduction</a> ·
  <a href="#real-robot">Real Robot</a> ·
  <a href="#repository-layout">Repository Layout</a>
</p>

> **Anonymous ICLR 2027 submission.** This repository is a public development
> mirror. Do not link it from a double-blind submission; use an anonymized
> artifact snapshot for review.

## Overview

Flow-matching robot policies conventionally begin action generation from an
uninformed Gaussian source. **Expert-Inversion Prior (EIP)** asks a different
question: *which source distribution should a frozen action generator start
from for the current observation?*

EIP first inverts expert action chunks through a frozen Action Flow. It then
learns an observation-conditioned Prior Flow that maps Gaussian noise to these
expert inversion sources. At test time, the learned source is decoded by the
unchanged Action Flow. The base policy, visual encoder, action normalizer, and
expert demonstrations remain fixed during Prior training.

This repository accompanies **The Source Matters: Understanding and Learning
Expert-Inversion Priors for Flow-Matching Robot Policies**. It contains the
current temporal-thinning method, the full-cache reference, paired simulation
rollouts, and the real-robot FM-UNet path for LeRobot v3 data.

## Method

For a frozen conditional Action Flow F_θ, offline expert inversion constructs an expert source z* for each action window and observation condition c. The conditional Prior Flow G_φ is trained with flow matching:

```text
z_t = (1 - t) ε + t z*
u   = z* - ε
L   = MSE(v_φ(z_t, t, c), u)
```

At rollout:

```text
observation -> frozen Action Flow encoder -> condition c
Gaussian ε -> Prior Flow G_φ -> source z_prior
(z_prior, c) -> frozen Action Flow F_θ -> action chunk
```

### Temporal thinning

Overlapping action windows create many nearly redundant inversion targets. The
active protocol retains every eighth window in each episode and always keeps
the terminal window:

```text
all windows:  0, 1, 2, 3, ..., final
stride = 8:  0, 8, 16, 24, ..., final
```

Prior training still uses a matched optimizer budget by cycling shuffled passes
through this smaller cache. The resulting comparison changes the source data,
not the base action policy or rollout decoder.

## Reproduction

The code is a patch layer over a compatible MomentVLA / RoboVerse checkout.
Datasets, checkpoints, expert-inversion caches, and rollout videos are not
included.

```bash
# Build a stride-8 cache from a full expert-inversion cache.
python prior_policy/temporal_thinning/build_stride8_cache.py \
  --source-cache /path/to/full_inversion_cache \
  --output-cache /path/to/stride8_inversion_cache \
  --stride 8

# Train the conditional Prior Flow for a fixed 37,500-step budget.
python prior_policy/temporal_thinning/train_stride8_prior.py \
  --repo /path/to/MomentVLA-main \
  --cache /path/to/stride8_inversion_cache \
  --output-dir /path/to/outputs/prior_stride8 \
  --action-flow-checkpoint /path/to/action_flow.ckpt \
  --device cuda:0 --seed 42 \
  --epochs 150 --batch-size 32 --max-train-steps 250 --train-all

# Run paired evaluation with P8+A10.
python prior_policy/runtime/prior_flow_rollout.py \
  --method prior \
  --action-checkpoint /path/to/action_flow.ckpt \
  --prior-checkpoint /path/to/outputs/prior_stride8/best.pt \
  --zarr-path /path/to/task.zarr \
  --output-dir /path/to/rollout_output \
  --task stack_cube --robot franka --sim isaacsim \
  --prior-steps 8 --action-steps 10 \
  --environment-seed 43 --noise-seed 0 --max-demos 50 --max-steps 300
```

Use the task manifests to reproduce the standardized settings:

| Task | Cache | Prior/action steps | Optimizer updates | Configuration |
|---|---:|---:|---:|---|
| PickCube | stride = 8 | P8+A10 | 37,500 | [config](configs/pickcube_stride8_p8a10.yaml) |
| StackCube | stride = 8 | P8+A10 | 37,500 | [config](configs/stackcube_stride8_p8a10.yaml) |
| CloseBox | stride = 8 | P8+A10 | 37,500 | [config](configs/closebox_stride8_p8a10.yaml) |

For paired comparisons, keep the Action Flow checkpoint, task state,
scene configuration, environment seed, source-noise schedule, and trial count
fixed across methods. Full commands and reporting rules are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Current simulation observations

The release records only results whose cache construction, update budget, and
paired rollout settings are specified in code. For example, the matched
7,500-step StackCube comparison pooled over seeds 42 and 43 produced 86/100
successes with the stride-8 cache and 78/100 with the full cache. The CloseBox
stride-8 P8+A10 run reached 30/50 after 37,500 updates, while its 7,500-step
run did not improve over the FM baseline. These are empirical observations,
not significance claims; see [the experimental record](theory/experimental_observations.md)
for trial-level context and limitations.

## Real robot

The [real_robot/](real_robot/README.md) directory provides a complete path from
LeRobot v3 data to FM-UNet and Prior Flow deployment:

1. convert and validate LeRobot v3 episodes;
2. train a frozen FM-UNet baseline;
3. build the temporally thinned expert-inversion cache;
4. train the conditional Prior Flow;
5. verify training/deployment input equivalence before connecting hardware.

The current two-view FM-UNet runtime uses synchronized 128×128 RGB histories,
8D joint/gripper states, and 8D joint/gripper targets. It loads the checkpoint
configuration, EMA state, LeRobot-DP ResNet, and saved normalizer together. The
verification script asserts that Zarr preprocessing and deployment inference
match exactly.

## Repository layout

```text
Inversion_policy/
├── configs/                         # task-level standardized settings
├── docs/                            # setup and reproduction protocol
├── prior_policy/
│   ├── full_data/                   # full-cache reference implementation
│   ├── temporal_thinning/           # active stride-8 cache and prior training
│   └── runtime/                     # frozen Action Flow and paired rollout
├── real_robot/                      # LeRobot v3 conversion and deployment
│   ├── conversion/                  # dataset adapter and validation
│   ├── deployment/                  # checkpoint runtime and input checks
│   ├── patches/                     # EMA and runner fixes for MomentVLA
│   └── scripts/                     # FM, cache, and prior entry points
├── theory/                          # theory, ablations, and result records
└── legacy/                          # superseded exploratory implementations
```

The active implementation is under [prior_policy/](prior_policy/README.md).
The full-cache code is retained as a reference rather than mixed with the
stride-8 protocol. `legacy/` is excluded from current training and evaluation.

## Citation

A citation entry and archival release will be added after the review process.

## License and data

This repository distributes source code and documentation only. Use the
licenses of the underlying MomentVLA / RoboVerse project and task assets. Do
not redistribute checkpoints, demonstrations, inversion caches, or videos
without the corresponding permissions.
