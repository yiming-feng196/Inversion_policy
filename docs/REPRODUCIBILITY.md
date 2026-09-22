# Reproducibility Guide

## LIBERO snapshot (2026-09-22)

The running 15-task LIBERO-90 protocol is documented separately in
[`prior_policy/libero90/README.md`](../prior_policy/libero90/README.md), with
server source hashes and observed environments in its
[`SNAPSHOT_PROVENANCE.json`](../prior_policy/libero90/SNAPSHOT_PROVENANCE.json).
It uses train-only normalization, disjoint episode splits, 10,000 base updates,
5,000 prior updates, RK4-256 inversion, Euler-16 source sampling and Midpoint-10
action decoding. Active bases are FM-UNet and FM-DiT; DP is paused and pi0 deferred.
Do not apply the older `--train-all`, 37,500-update or P8 settings below to the
LIBERO results. The following sections describe the separate original protocol.

## 1. Produce or obtain a full expert-inversion cache

The current release supports direct temporal thinning from a full cache whose shards contain at least:

```text
sample_index, sampler_index, split, episode, condition, z_star
```

The `split` and `episode` assignments are copied unchanged into the thinned cache.

## 2. Derive the stride=8 cache

```bash
python prior_policy/temporal_thinning/build_stride8_cache.py \
  --source-cache /path/to/full_inversion_cache \
  --output-cache /path/to/stride8_inversion_cache \
  --stride 8
```

For each episode and split, the script sorts by `sampler_index`, keeps every eighth window, and adds the terminal window. It copies existing cached tensors rather than rerunning Action Flow inversion.

## 3. Train with the unified update budget

The thinned cache has fewer natural minibatches than the full cache. Specify the desired update budget through `epochs` and `max-train-steps`:

```bash
python prior_policy/temporal_thinning/train_stride8_prior.py \
  --repo /path/to/MomentVLA-main \
  --cache /path/to/stride8_inversion_cache \
  --output-dir /path/to/outputs/prior_stride8 \
  --action-flow-checkpoint /path/to/30.ckpt \
  --device cuda:0 --seed 42 \
  --epochs 150 --batch-size 32 --max-train-steps 250 \
  --prior-inference-steps 16 --train-all
```

All reported temporal-thinning runs use `batch_size=32`, `150 x 250 = 37,500` optimizer updates, and `--train-all`. The trainer cycles fresh shuffled passes through the thinned cache inside an epoch when needed. With `--train-all`, the cache validation split is used only as an in-distribution diagnostic; it is not a held-out generalization metric.

## 4. Evaluate P8+A10

```bash
python prior_policy/runtime/prior_flow_rollout.py \
  --method prior \
  --action-checkpoint /path/to/30.ckpt \
  --prior-checkpoint /path/to/outputs/prior_stride8/best.pt \
  --zarr-path /path/to/task.zarr \
  --output-dir /path/to/rollout_output \
  --task stack_cube --robot franka --sim isaacsim \
  --prior-steps 8 --action-steps 10 \
  --environment-seed 43 --noise-seed 0 \
  --max-demos 50 --max-steps 300
```

For paired comparisons, hold fixed the task, action checkpoint, trajectory source, scene settings, environment seed, start demo, number of demos, and source-noise schedule.

## Experiment configurations

- [`configs/stackcube_stride8_p8a10.yaml`](../configs/stackcube_stride8_p8a10.yaml)
- [`configs/closebox_stride8_p8a10.yaml`](../configs/closebox_stride8_p8a10.yaml)

## Reporting

Report successes and trials for every seed, inference time, exact prior/action step counts, cache selection rule, optimizer updates, Action Flow checkpoint hash, and paired rollout conditions. The exploratory results are summarized in [`theory/experimental_observations.md`](../theory/experimental_observations.md).
