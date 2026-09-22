# LIBERO expert-source priors: lab snapshot, 2026-09-22

This directory contains the **actual running LIBERO experiment implementation**:
FM base-policy training, stride-8 expert inversion, conditional Gaussian/Flow
source training, preprocessing checks, and paired closed-loop evaluation.
Runtime files were copied from the lab server without changing the training or
inference implementation. SHA256 provenance is in
[`SNAPSHOT_PROVENANCE.json`](SNAPSHOT_PROVENANCE.json).

## Active scope

- 15 selected LIBERO-90 tasks; **not the full LIBERO-90 benchmark**.
- Active bases: **FM-UNet and FM-DiT**. Each task has its own base policy.
- Methods: standard Gaussian initialization (G), conditional diagonal Gaussian
  (CG), and conditional Flow (CF).
- **DP/Diffusion was paused by explicit user request on September 22.** Its code
  is retained for diagnosis; do not interpret it as a validated extension or
  resume its queue without authorization. Pi0 remains deferred.
- Active denominator: 30 task/base combinations and 10,500 rollouts.

The separate [π₀ two-task LIBERO-90 requirements](pi0/TWO_TASK_REQUIREMENTS.md)
specify stride-8 experiments on task 35 and candidate task 8, including baseline
confirmation before expensive inversion. This is a requirements-only plan, not
an activation of the π₀ queue; task 8's π₀ baseline is not yet established.

Task order is fixed in [`active_scope.json`](active_scope.json): original five
IDs **35, 13, 8, 44, 73**, followed by **0, 16, 19, 21, 27, 33, 46, 62, 65, 86**.
[`tasks.json`](tasks.json) maps the official LIBERO-90 IDs to full task names.
The original and extension cohorts should be reported separately.

## Protocol (do not mix with the repository's IsaacSim configuration)

| Setting | LIBERO snapshot |
|---|---|
| Episode split | 35 train / 5 validation / 10 test; split seed 20260915 |
| Normalization | Train-only action/proprio statistics; frozen checkpoint normalization |
| Observations | Two frames, two RGB views, proprioception |
| Action chunk | Complete 16 x 7; execute indices `[1:9]` |
| Base training | 10,000 updates, batch 32, seed 0; validation-objective checkpoint |
| Expert inversion | FP32 RK4-256; select windows **before** inversion |
| Thinning | Every eighth chronological window per episode plus final eligible window |
| Prior training | 5,000 updates, batch 32, seeds 0/1/2; fixed-final checkpoint |
| Prior/decoder integration | CF Euler-16; frozen Action Flow Midpoint-10 |
| Evaluation | 50 matched initial states; environment seed 7; 10 settling steps; 400-step limit |

G is evaluated once per frozen checkpoint. CG/CF each use three prior-training
seeds with 50 rollouts per seed, giving 350 rollouts per task/base combination.
Base-policy training seeds are not replicated. Simulator initial states are
matched across methods, not certified unseen demonstration layouts.

The older P8+A10 / 37,500-update / `--train-all` configuration elsewhere in this
repository is a **different experiment protocol**. It is not used by this queue.

## Files

- `fm_adapter/libero_fm.py`: audited LIBERO dataset adapter, vision encoder,
  author UNet/DiT backbones, FM training and ODE integration.
- `fm_adapter/q1_statistics.py`: inversion/distribution diagnostics.
- `extract_cache.py`, `thinning.py`: deterministic, resumable expert-source
  extraction with checkpoint, normalization and dataset hashes.
- `training/source_models.py`: conditional Gaussian, conditional/unconditional
  Flow and MLP source models; `train_samplers.py` trains them on frozen sources.
- `evaluation/preflight_libero.py`: observation/action convention checks.
- `evaluation/eval_closed_loop.py`: matched rollouts with a fixed decoder.
- `prepare_data.py`, `audit_frs_tasks.py`: pinned dataset validation and FRS task
  reconciliation. This is not an official reproduction of FRS training.
- `run_queue.py`, `continue_extension.py`, `collect_status.py`: cooperating
  GPU workers, bounded continuation and read-only status collection.
- `dp_adapter/`, `pi0/`: retained **inactive** implementations.

`SERVER_PROTOCOL_HISTORY.md` and `HISTORICAL_STATUS_20260921.md` preserve earlier
server documentation, including superseded full-suite/five-task plans. They do
not override `active_scope.json` or the September 22 runbook revision.
`switch_to_five.py`, `pause_robocasa.py`, and `fm_adapter/run_queue.py` are historical
lab orchestration helpers, **not current entry points**.

## Environment and portability

This is a source snapshot, not a self-contained simulator installation. A
compatible MomentVLA/RoboVerse checkout supplies the author backbones; official
LIBERO assets, HDF5 demonstrations and simulation dependencies are external.
The observed training and simulation package versions and backbone hashes are
recorded in `SNAPSHOT_PROVENANCE.json`. No credentials, demonstrations, weights,
latent caches, simulation records, videos or third-party dependency trees are
included.

Core stage scripts accept explicit paths. For example, from the repository root:

```bash
python prior_policy/libero90/fm_adapter/libero_fm.py train \
  --repo /path/to/MomentVLA --arch unet --hdf5 /path/to/task_demo.hdf5 \
  --output /path/to/artifacts/base --seed 0 --steps 10000 \
  --batch-size 32 --eval-every 500 --validation-samples 256

python prior_policy/libero90/extract_cache.py \
  --adapter-dir prior_policy/libero90/fm_adapter --repo /path/to/MomentVLA \
  --checkpoint /path/to/artifacts/base/best.pt --output /path/to/artifacts/cache \
  --stride 8 --inverse-steps 256 --solver rk4 --decoder midpoint --decoder-steps 10

python prior_policy/libero90/training/train_samplers.py \
  --cache /path/to/artifacts/cache --repo /path/to/MomentVLA \
  --output /path/to/artifacts/cflow_s0 --method cflow --seed 0 \
  --steps 5000 --batch-size 32 --prior-steps 16
```

Use `--method cgaussian` and a distinct output directory for CG. Retain all three
predeclared seeds. Run the preprocessing gate before closed-loop evaluation;
the exact launch arguments and hashes are recorded by the queue. Use each stage's
`--help` for required paths. Do not bypass checkpoint/cache/normalization checks.

The **lab scheduler intentionally retains its original absolute paths**. Before
running it elsewhere, configure the interpreter, adapter, simulator, author-repo,
dataset and LIBERO-config paths in a separate deployment copy. In particular,
the bundled `fm_adapter/` corresponds to the server's
`q1_libero_crossarch_20260915/code_v2_20260916/` directory. Do not run a second
scheduler against an already active experiment root. Copying this source snapshot
does not modify or restart the existing lab queue.

## Lightweight verification

```bash
cd prior_policy/libero90
python -m unittest test_extension test_thinning
python -m compileall -q .
```

These 13 lightweight tests cover task scope, disabled DP dispatch, locking,
explicit failure handling and temporal thinning. They are not an end-to-end
robotics validation. `fm_adapter/test_libero_fm.py` additionally requires PyTorch
and `Q1_AUTHOR_REPO` and is run from the `fm_adapter/` directory.

DP negative results remain on the server. Their cause has not been established;
pausing DP is a workload decision, not a conclusion that the implementation is
necessarily wrong or that the negative outcomes can be discarded.
