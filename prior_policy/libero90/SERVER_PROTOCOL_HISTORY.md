# LIBERO-90 temporal-thinning evaluation — 2026-09-21

## Active scope changed: five tasks, no pi0

The user's latest instruction supersedes the original full-suite plan below.
`active_scope.json` now restricts dispatch to task IDs **35, 13, 8, 44, 73** and
**FM-UNet / FM-DiT / DP-UNet**. Pi0 and the other 85 tasks are deferred.
The same stride-8-plus-terminal selection, full chunks, source seeds 0/1/2,
and 50 paired trials remain unchanged. This is a five-task pilot, not a
LIBERO-90 full-suite average or a randomly sampled representative benchmark.

Tasks: microwave opening; front bowl onto plate; open drawer then place bowl;
turn on stove; book into the front caddy compartment. Selection is based on
interaction types and reuse of the already-running microwave task, not on
comparative success rates. All three bases use exactly the same task set.

`switch_to_five.py` freezes only the old scheduler processes, allows useful
active GPU stages to complete, then gracefully restarts the restricted queues.
All existing checkpoints, rollouts, inverse caches and partial downloads are
preserved. Only these five datasets are prioritized for further downloading.
The archived original configuration below documents provenance, not the active
workload. Status scripts must use `active_scope.json` as their denominator.

## Scope approved by the user

Full **LIBERO-90**, not the 15-task online-DSBC or 62-task zero-shot subset.
Reconcile FRS's 89 success-filtered tasks separately; do not select exclusions
based on our results. RoboCasa GPU workers were paused with all completed
inversion files preserved. Other users' GPU processes were not touched.

Four base-policy families: FM-UNet, FM-DiT, DP-UNet with deterministic DDIM,
and frozen `pi0_libero`. Each compares standard Gaussian sources, conditional
diagonal Gaussian priors, and conditional Flow priors. FM and DP base models
are trained per task with the same data split, not a shared language-conditioned
90-task backbone. Pi0 is the existing adapted checkpoint, not a newly fine-tuned
LIBERO-90 checkpoint. Do not call these official FRS model reproductions.

## Temporal thinning

Reference user repository commit:
`bd132c37d32c868d1df531bd6b3fbea01a45a65d` (`origin/main`, fetched without changing
the user's local branch). Selection matches
`prior_policy/temporal_thinning/build_stride8_cache.py`:
within each split and episode, sort chronologically, take `[::8]`, retain the
final window once. Keep the entire action/source chunk, unchanged.

`extract_cache.py` and `pi0/extract.py` select **before inversion**. This is not
fewer integration steps, source-coordinate interpolation, or shorter chunks.
Existing complete caches can be subset exactly with `derive_cache.py`; doing
so does not retrospectively save the inversion time already spent.

## Fixed protocol

- Public raw HDF5 release: `yifengzhu-hf/LIBERO-datasets`, pinned revision
  `f13aa24a3da8c43c7225569f28c562979fa0e35a`. Every file must match its published
  LFS SHA256 and size before entering the queue. LIBERO-90 contains 90 files,
  66,658,085,995 bytes. Existing microwave file is reused only after validation.
- `tasks.json` is generated from the installed official LIBERO benchmark.
- FM/DP: disjoint 70/10/20 episode split (normally 35/5/10), seed 20260915.
  Train-only min/max action and proprio normalization. Two observation frames,
  two RGB cameras, complete 16x7 chunks, execute `[1:9]` (8 steps).
- Base FM: existing A2A author velocity backbones with our audited LIBERO
  adapter; ResNet18+GroupNorm from scratch. 10,000 updates, batch 32, FP32,
  TF32 disabled; choose base checkpoint by validation objective, not rollout.
- FM sources: RK4-256 inversion, frozen Midpoint-10 deployment.
- DP: same UNet/vision architecture and 10,000-update budget, epsilon MSE;
  official diffusers 0.35.1 schedulers, 100-step cosine training schedule.
  DDIM-100 eta=0 for ALL three deployment methods; inverse DDIM-100 for targets.
  `clip_sample=True`. Inversion is approximate, and clipping is not invertible.
  This is **DP-DDIM**, not an unchanged stochastic-DDPM inference baseline.
  No EMA in this matched-budget adapter. Legacy training `*_fm` log fields mean
  epsilon MSE for DP; cache `midpoint10_rmse` field means deployed-decoder RMSE,
  whose actual scheduler and step count are explicitly stored in the manifest.
- Pi0: frozen existing `pi0_libero`, complete 50x32 sources, 7 active channels;
  RK4-1280 inversion, native-equivalent FP32 Euler-10 action decoder, execute 10.
  Retain the checkpoint's existing normalization. Split seed 20260919 preserves
  the earlier pi0 protocol. Source condition is pooled frozen prefix features
  plus normalized state, never expert future actions or episode IDs.
  Terminal repeat-last padding remains explicit in `valid_steps`.
- Both prior methods: identical selected train windows, 5,000 optimizer updates,
  batch 32, learning rate 1e-4, seeds 0/1/2, fixed-final checkpoint. Flow prior
  uses Euler-16. This deliberately preserves the earlier audited LIBERO budget
  for before/after comparisons; it does NOT combine thinning with the user
  repository's separate P8 / 37,500-update configuration change.
- Every method: 50 packaged simulator initial states, env seed 7, 10 settling
  steps, maximum 400 task steps on LIBERO-90. Gaussian baseline is evaluated
  once per base checkpoint; repeated prior-training seeds are not independent
  Gaussian baselines. Initial states are matched, not certified unseen layouts.
- Preprocessing convention check must pass before FM/DP rollouts. Pi0 retains
  numerical parity, normalization roundtrip and inversion convergence gates.
- Success rate is computed from complete rollout records only. Do not report
  partially evaluated tasks as zero or silently exclude failures.

## Before/after comparisons

- `KITCHEN_SCENE7_open_the_microwave` is task 35 in LIBERO-90. Existing pi0
  sources used **stride 10**, NOT dense stride 1; compare honestly as stride 10
  versus stride 8 plus terminal, not full versus thinned.
- Previous `OpenDrawer` and `BowlOnPlate` are LIBERO-Goal tasks. Their four
  full caches are retained and exact stride-8 subsets were prepared under
  `controls/goal_*`. They are useful matched-checkpoint thinning controls but
  must not be counted toward LIBERO-90 averages. Additional prior training and
  rollout evaluation are required before calling these comparisons complete.
- FRS uses success-filtered TFDS and a different base VLA. Its 89 tasks were
  reconciled against the author's 3,917-demo reasoning index, revision
  `52e8a9caf92b81d665af679fac760f7740e3e3aa`: the missing task is **ID 51**,
  `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket`.
  Main evaluation retains all 90 tasks. A secondary common-89 mean excludes
  task 51 by this external list, not by our performance. Our raw-HDF5 study
  remains different from FRS's success-filtered TFDS data/model protocol.
  `frs_audit/` preserves source hashes and successful-demo IDs; no filtering is
  silently applied to the running training experiment.

## Server execution

Root: `/data/jhr/libero90_thinning_20260921`.
`prepare_data.py` downloads two files concurrently with resumable `.part` files.
`run_queue.py --gpu 0 --first unet` and `--gpu 1 --first dit` cooperate via task
locks and write `worker_gpu{0,1}.json`. Each task family performs base training
if needed, thinned inversion, prior training and paired rollouts. Errors are
recorded and stop the affected worker instead of producing misleading scores.
Task job statuses live in `jobs/`; stage logs in `logs/`; rollout results in
`rollouts/`. Pi0 and DP are explicitly queued, not claimed running until their
stage is visible in the worker status.

Validation performed: five thinning tests; DP two-step training and 2-window
per-split DDIM extraction smoke; DP import in the simulator environment.
`test_dp_scheduler.py` additionally checks scheduler ordering with a zero-epsilon
analytic field. This test does not establish exact inversion for learned DP.

The two-GPU full-suite run is a long-running workload, not an overnight
completion promise. Progress and completed subset counts must accompany all
interim tables. Timing measured with competing jobs is not a clean latency
benchmark.
