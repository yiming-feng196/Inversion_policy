# LIBERO-90 fifteen-task continuation, 2026-09-21

## User-authorized revision, 2026-09-22

The user explicitly paused all Diffusion/DP experiments. Active families are now
FM-UNet and FM-DiT only: 15 tasks x 2 families = 30 configurations and 10,500
rollouts. DP results, checkpoints, caches, logs and incomplete rollouts remain
preserved and must not be deleted or relabeled successful. Do not resume DP
without a new explicit user request. A code defect is suspected, not established.
Pi0 stays deferred. All FM scientific settings and the 15-task order are unchanged.

The two DP worker processes were gracefully stopped, along with their dispatcher,
before revising the scope. Deployment snapshots and the new activation record are
under pause_dp_20260922/. The revised supervisor dispatches only unet,dit. The
original extension_20260921/activation.json is historical; use the new activation
record for current scheduler/scope hashes. collect_status.py now reports active
FM configurations only; the DP artifacts remain accessible in their old folders.

The 24-hour monitoring window is NOT restarted by this revision. Its original
start was 2026-09-21 21:58 CST; report and stop monitoring around 2026-09-22
21:58 CST without stopping healthy FM workers.

The following original activation description is retained as historical context;
where it mentions active DP or 45 configurations, this revision takes precedence.

## Authority and frozen protocol

The user requested appending more tasks to the existing queue and running
experiments continuously while they are away for the next 24 hours. The original
five task IDs stay first: 35, 13, 8, 44, 73. Added IDs are fixed before results:
0, 16, 19, 21, 27, 33, 46, 62, 65, 86. This is not the full LIBERO-90 suite.

Use FM-UNet, FM-DiT, DP-UNet; Gaussian baseline, conditional Gaussian, conditional
Flow; stride 8 plus terminal eligible window; complete 16 x 7 chunks; 35/5/10
episode split seed 20260915; train-only statistics; 10,000 base updates; 5,000
prior updates; prior training seeds 0/1/2; 50 paired evaluation states; the current
400-step horizon and decoder settings. Pi0 remains deferred. Do not change
scientific settings, add further tasks, select models on success, or delete
negative/incomplete runs during unattended supervision.

15 tasks x 3 base families = 45 model-task configurations. A complete configuration
has 350 rollouts (G: 50 once; CG/CF: 3 x 50 each), or 15,750 across the full selected
scope. Original five and added ten are separate predeclared cohorts.

## Server

- Host: jhr@192.168.100.101 (existing user-authorized connection; no password in files).
- Root: /data/jhr/libero90_thinning_20260921
- Python: /data/jhr/q1_libero_crossarch_20260915/venv/bin/python
- Sim Python: /data/jhr/q2_sampler_comparison_20260919/sim_venv/bin/python

The existing worker PIDs at activation are 1504680 (GPU0), 1503513 (GPU1).
They are not interrupted. They retain their original five-task in-memory list.
The detached continue_extension.py supervisor waits for each worker lock to become
free, checks for surviving stage children, and starts the expanded queue. Original
completed jobs are skipped. A new --continue-on-error scheduler flag preserves
failed statuses and processes unrelated jobs; it does not disguise failure as
success or automatically reset failed jobs. Launch retries are bounded to three
per GPU. Missing official datasets download concurrently with pinned hash checks.

The finite queue keeps running independently of the laptop/SSH session until
finished or explicitly stopped. The 24-hour app heartbeat is an additional audit,
not the mechanism keeping the GPU jobs alive. Do not stop healthy jobs just because
24 hours have elapsed; report the snapshot to the returning user.

## Read-only checks

Run collect_status.py and inspect extension_20260921/supervisor.json,
extension_20260921/activation.json, worker_gpu0.json and worker_gpu1.json.
Verify the reported PIDs and the newest stage logs before claiming progress.
Use nvidia-smi for process/utilization checks, not benchmark latency measurements.

Expected state immediately after activation: both original workers still running,
zero expanded-worker launches, supervisor running, and at most one data downloader.
After original completion, expanded-worker launches appear in supervisor events.

## Recovery boundaries

- If a worker exits, inspect its stage log and job status; preserve completed data.
- Failed jobs are explicit and skipped by expanded workers so other work continues.
- Only retry a failed job after diagnosing the cause and recording a recovery note.
- Resume supported checkpoints/rollouts with their original settings. No data
  deletion, normalization changes, hyperparameter changes, or other users' kills.
- A preflight/data-hash/numerical-NaN issue requires inspection, not an automatic
  gate bypass. Unknown errors require a user-visible diagnostic.
- After an SSH/network failure do not assume the remote experiment stopped.
- The existing detached supervisor is the only dispatcher; don't launch another
  unless its process and lock are both confirmed absent. It uses its own lock.
- If the whole queue finishes early, report completion rather than spending GPU
  time on unapproved tasks or rerunning completed work.

To stop only further supervisor dispatch, create extension_20260921/stop_dispatch.flag.
Already running workers are intentionally untouched; stopping GPU work requires
separate verified process-level handling at the user's request.

## Audit artifacts

Original code/scope files are preserved in extension_20260921/original_files.
activation.json records before/after hashes and the supervisor PID.
No model-training or numerical-inversion code is changed by this extension.
Unit tests cover scope/order, protocol freeze, locking, explicit failures and
existing temporal-thinning behavior. Results must retain all predeclared tasks,
all three prior seeds, and any degradation or failure.
