# Active five-task scope — 2026-09-21, 18:52 CST

This scope supersedes the original 90-task launch below. The user requested
deferring pi0 and evaluating the smaller models on five shared tasks.

- Tasks: 35 (microwave), 13 (front bowl to plate), 8 (open drawer and place bowl),
  44 (turn on stove), 73 (book into front caddy compartment).
- Families: FM-UNet, FM-DiT, DP-UNet; Gaussian baseline, conditional Gaussian,
  and conditional Flow. Three prior-training seeds, 50 evaluation states each.
- All five selected datasets are downloaded and verified. Existing data,
  checkpoints, caches, and completed rollouts are preserved.
- Stride-8 selection occurs before inversion, per episode, retaining the final
  window and the complete action chunk. Microwave: 6,664 eligible windows become
  896 selected windows (620 / 90 / 186 train / validation / test).
- GPU1 worker 1503513 is running the restricted queue; DP microwave base training,
  inversion, and preflight have finished, and conditional Gaussian training has begun.
- GPU0 finished its last microwave UNet evaluation (100 / 100) and switched to
  restricted worker 1504680. Both workers now run only the selected five tasks
  and three smaller model families. The scope-switch completion journal is saved.
- No pi0 job was started by this suite; it is excluded from the active queue.
- Report this as a five-task pilot, not a LIBERO-90 full-suite average. Selection
  is based on task mechanics and reuse, not comparative method performance.

Use `collect_status.py`, `active_scope.json`, and `scope_switch_20260921/` for
live state. The data-scope completion record confirms 5 / 5 verified datasets.
Two old downloader curl children were identified by exact dataset paths and
terminated; their partial files remain available for later resumption.

## Archived original launch snapshot (superseded)

RoboCasa workers 1464000 / 1464001 were stopped by exact executable identity
checks, preserving all completed shards. Their statuses are `paused_user_request`.

LIBERO workers: GPU0 PID 1480816; GPU1 PID 1480817. Both are live detached
queues with 90 tasks x 4 policy families, and source-prior seeds 0/1/2.
Latest inspected stages:

- GPU0: microwave task 35, FM-UNet paired closed-loop evaluation, seed 0.
- GPU1: same task, FM-DiT paired closed-loop evaluation, seed 0.
- Both FM base models completed 10,000 updates; both stride-8 source caches
  complete: 620 / 90 / 186 train / val / test windows, versus
  4,612 / 679 / 1,373 eligible dense windows. Whole chunks are retained.
- Both architectures' seed-0 conditional Gaussian and conditional Flow training complete.
- 9 / 90 official HDF5 files verified at the last snapshot; downloads continue.
  A truncated CDN transfer was resumed; explicit retries for curl error 18 were
  added, with partial files preserved and mandatory final SHA256 verification.
- DP training/inversion smoke and two analytic DDIM tests passed; production
  DP and pi0 stages are queued, not yet reported as running or complete.
- Four existing LIBERO-Goal thinning-control caches have been prepared;
  their new prior training / rollout comparisons are still pending.

No final success-rate claim: the first DiT evaluation is still incomplete.
Use `collect_status.py` on the server for current state, not this static note.

FRS task reconciliation: author release contains 89 tasks / 3,917 demo IDs.
Missing task = 51, `LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket`.
Main run retains all 90; a common-89 aggregate is secondary. Raw-HDF5 training
is not silently relabeled as FRS's success-filtered TFDS protocol.
