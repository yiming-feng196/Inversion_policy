"""Paired LIBERO rollouts for fixed source samplers and a frozen Q1 Action Flow.

Imports simulator and GPU libraries only in main(). Use --trials 2 for a smoke
test. This script does not train, select checkpoints using rollouts, or claim
that packaged initial-state indices correspond to held-out demonstration IDs.
"""
from __future__ import annotations

import argparse
import collections
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from libero_eval_utils import (raw_frame, policy_inputs, executable_actions,
                              paired_noise_seed, paired_success_summary)


LEARNED_METHODS = {"mlp", "cgaussian", "cflow", "uflow"}
METHODS = ("gaussian", "zero", "train_cond_nn", "train_nn", "mlp", "cgaussian", "cflow", "uflow")


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_sampler_paths(items):
    paths = {}
    for item in items:
        method, path = item.split("=", 1)
        if method not in LEARNED_METHODS or method in paths:
            raise ValueError(f"Invalid or repeated learned sampler: {method}")
        paths[method] = Path(path).resolve()
    return paths


def resolve_task(suite, task_name):
    matches = [(index, suite.get_task(index)) for index in range(suite.n_tasks)
               if suite.get_task(index).name == task_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact task.name={task_name!r}; found {len(matches)}")
    return matches[0]


@contextmanager
def trusted_packaged_state_loading(torch):
    # Scope legacy loading strictly to LIBERO's installed initial-state asset.
    original = torch.load
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def state_digest(value):
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(str(arr.dtype).encode() + repr(arr.shape).encode() + arr.tobytes()).hexdigest()


def case_key(record):
    return (record["task"], record["initial_state_index"], record["env_seed"], record["policy_seed"], record["method"])


def visual_keys(condition):
    # Each chronological frame is [camera0_512, camera1_512, state8].
    if condition.ndim != 2 or condition.shape[1] != 2064:
        raise ValueError("This adapter requires frozen 2-frame, 2-camera conditions of dimension 2064")
    keys = condition.reshape(-1, 2, 1032)[:, :, :1024].reshape(len(condition), -1)
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    if not np.isfinite(keys).all() or np.any(norms < 1e-12):
        raise ValueError("Nonfinite or zero visual retrieval key")
    return keys / norms


def standardized_condition_keys(train_condition):
    """Train-only per-coordinate statistics, same convention as learned models."""
    if train_condition.ndim != 2 or train_condition.shape[1] != 2064 or not np.isfinite(train_condition).all():
        raise ValueError("Require finite complete training conditions")
    values = train_condition.astype(np.float64)
    mean = values.mean(0).astype(np.float32)
    std = np.maximum(values.std(0), 1e-5).astype(np.float32)
    return (train_condition - mean) / std, mean, std


def validated_preflight(path, *, task_name, data_sha256):
    """Optional external gate; deployment queue should always supply it."""
    if path is None:
        return {"provided": False, "scope": "No preprocessing preflight supplied to runner"}
    result = json.loads(Path(path).read_text())
    if result.get("status") != "complete" or result.get("preprocessing_gate_passed") is not True:
        raise ValueError("Preprocessing preflight must be complete and pass its gate")
    if result.get("args", {}).get("task_name") != task_name:
        raise ValueError("Preflight task does not match rollout task")
    if result.get("data_sha256") != data_sha256:
        raise ValueError("Preflight did not use the frozen action policy's exact dataset")
    return {"provided": True, "path": str(Path(path).resolve()), "sha256": digest_file(path),
            "data_sha256": data_sha256, "task_name": task_name,
            "gate_mode": result.get("gate_mode", "legacy"),
            "encoding_gate_metrics": result.get("encoding_gate_metrics"),
            "state_max_abs_error": result["state_max_abs_error"],
            "image_comparisons": result["image_comparisons"]}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--q1-code", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--training-code", type=Path, default=Path(__file__).resolve().parents[1] / "training")
    p.add_argument("--action-checkpoint", type=Path, required=True)
    p.add_argument("--cache-manifest", type=Path, required=True)
    p.add_argument("--preflight-report", type=Path,
                   help="Passed preflight_libero.py JSON; queue should provide this before learned rollouts")
    p.add_argument("--sampler", action="append", default=[], metavar="METHOD=FINAL_PT")
    p.add_argument("--methods", default="gaussian,zero,train_cond_nn,mlp,cgaussian,cflow,uflow")
    p.add_argument("--task-name", required=True)
    p.add_argument("--suite", default="libero_90", choices=["libero_goal", "libero_90"])
    p.add_argument("--initial-state-start", type=int, default=0)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--env-seed", type=int, default=7)
    p.add_argument("--policy-seeds", default="0")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--wait-steps", type=int, default=10)
    p.add_argument("--decoder", choices=["midpoint", "euler", "rk4", "ddim"], default="midpoint")
    p.add_argument("--decoder-steps", type=int, default=10)
    p.add_argument("--prior-steps", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-allocator-mib", type=int, default=6144)
    p.add_argument("--clip-env-actions", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    methods = args.methods.split(",")
    seeds = [int(item) for item in args.policy_seeds.split(",")]
    if len(methods) != len(set(methods)) or not methods or any(m not in METHODS for m in methods):
        raise ValueError("Supply nonempty distinct supported methods")
    if (args.trials < 1 or args.initial_state_start < 0 or args.wait_steps < 2 or args.max_steps < 1
            or args.decoder_steps < 1 or args.prior_steps < 1 or len(set(seeds)) != len(seeds)
            or not seeds or min(seeds) < 0):
        raise ValueError("Invalid evaluation schedule")
    sampler_paths = parse_sampler_paths(args.sampler)
    if set(sampler_paths) != set(methods).intersection(LEARNED_METHODS):
        raise ValueError("Provide exactly one --sampler METHOD=PATH for each requested learned method")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "rollouts.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError("Results already exist; use --resume with identical protocol")
    sys.path.insert(0, str(args.q1_code.resolve()))
    sys.path.insert(0, str(args.training_code.resolve()))
    import torch
    from libero_fm import Policy, integrate
    from protocol import validate_manifest
    from source_models import load_sampler
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
        device_index = torch.device(args.device).index or 0
        total = torch.cuda.get_device_properties(device_index).total_memory
        torch.cuda.set_per_process_memory_fraction(min(args.max_allocator_mib * 1024**2 / total, 1.), device_index)

    cache = json.loads(args.cache_manifest.read_text())
    validate_manifest(cache)
    action_hash, cache_hash = digest_file(args.action_checkpoint), digest_file(args.cache_manifest)
    if action_hash != cache["action_checkpoint_sha256"]:
        raise ValueError("Action checkpoint does not match inverse-source cache")
    if cache["latent_shape"] != [16, 7] or cache["condition_dim"] != 2064 or cache["executed_slice"] != [1, 9]:
        raise ValueError("Unsupported action/condition/cache alignment")
    checkpoint = torch.load(args.action_checkpoint, map_location="cpu", weights_only=False)
    manifest = checkpoint["manifest"]
    if manifest["normalizer_fit_split"] != "train" or manifest["splits"] != cache["splits"]:
        raise ValueError("Action/source normalizer or episode split mismatch")
    normalizer = manifest["normalizer"]
    preflight = validated_preflight(args.preflight_report, task_name=args.task_name,
                                   data_sha256=manifest["data_sha256"])
    if digest_json(normalizer) != cache["action_normalizer_sha256"]:
        raise ValueError("Action normalization differs from the recorded source-cache normalization")
    base = Policy(args.repo, manifest["args"]["arch"], manifest["args"]["seed"]).to(args.device).eval()
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    base.requires_grad_(False)
    del checkpoint

    train_path = args.cache_manifest.parent / cache["files"]["train"]
    if digest_file(train_path) != cache["file_sha256"]["train"]:
        raise ValueError("Train source cache hash mismatch")
    with np.load(train_path, allow_pickle=False) as f:
        # Stable cache API shared with train_samplers.py; targets stay native.
        train_condition = f["condition"].astype(np.float32)
        train_source = f["z_star"].astype(np.float32)
        train_episode = f["episode"].astype(str)
    if len(train_condition) != len(train_source) or len(train_episode) != len(train_source):
        raise ValueError("Misaligned train bank")
    if set(train_episode).difference(cache["splits"]["train"]):
        raise ValueError("Non-training episodes in retrieval bank")
    if train_source.shape != (len(train_condition), 16, 7) or not np.isfinite(train_source).all():
        raise ValueError("Invalid native train sources")
    bank_keys = visual_keys(train_condition)
    full_bank_keys, retrieval_mean, retrieval_std = standardized_condition_keys(train_condition)

    samplers, sampler_info = {}, {}
    for method, path in sampler_paths.items():
        sampler, metadata = load_sampler(path, args.repo, args.device)
        if sampler.method != method or sampler.latent_shape != (16, 7) or sampler.condition_dim != 2064:
            raise ValueError("Wrong source sampler method/dimensions")
        # Exact metadata keys are written by this experiment's training pipeline.
        if metadata["manifest"]["cache_manifest_sha256"] != cache_hash:
            raise ValueError("Sampler trained on a different cache manifest")
        if (metadata["action_checkpoint_sha256"] != action_hash or
                metadata["action_normalizer_sha256"] != cache["action_normalizer_sha256"]):
            raise ValueError("Sampler trained for different frozen decoder/normalization")
        if (metadata["step"] != metadata["manifest"]["args"]["steps"] or
                metadata["checkpoint_selection"] != "fixed final optimizer step"):
            raise ValueError("Only the predeclared final-step sampler is eligible for closed-loop comparison")
        samplers[method] = sampler
        sampler_info[method] = {"path": str(path), "sha256": digest_file(path),
                                "method": method, "cache_manifest_sha256": cache_hash}

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_id, task = resolve_task(suite, args.task_name)
    with trusted_packaged_state_loading(torch):
        initial_states = suite.get_task_init_states(task_id)
    indices = list(range(args.initial_state_start, args.initial_state_start + args.trials))
    if indices[-1] >= len(initial_states):
        raise ValueError("Requested initial-state index exceeds packaged state count")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    schedule = [{"task": args.task_name, "initial_state_index": index,
                 "initial_state_sha256": state_digest(initial_states[index]),
                 "env_seed": args.env_seed, "policy_seed": seed}
                for index in indices for seed in seeds]
    shared = {"task": args.task_name, "suite": args.suite, "task_id": task_id,
              "bddl_sha256": digest_file(bddl), "action_checkpoint_sha256": action_hash,
              "cache_manifest_sha256": cache_hash, "action_normalizer_sha256": cache["action_normalizer_sha256"],
              "normalizer_canonical_json_sha256": digest_json(normalizer),
              "train_cache_sha256": cache["file_sha256"]["train"], "history": 2, "image_resolution": 128,
              "preprocessing_preflight": preflight,
              "horizon": 16, "executed_slice": [1, 9], "decoder": args.decoder,
              "decoder_steps": args.decoder_steps, "max_steps": args.max_steps,
              "wait_steps": args.wait_steps, "clip_env_actions": args.clip_env_actions,
              "precision": "FP32; TF32 disabled", "schedule": schedule,
              "retrieval_primary": "train_cond_nn: RMS distance in train-standardized full frozen condition",
              "retrieval_optional": "train_nn: cosine of visual features only",
              "base_adapter_sha256": digest_file(args.q1_code / "libero_fm.py"),
              "evaluation_code_sha256": {p.name: digest_file(p) for p in
                   (Path(__file__), Path(__file__).with_name("libero_eval_utils.py"))}}
    protocol_id = digest_json(shared)
    report = {"status": "running", "protocol_id": protocol_id, "protocol": shared,
              "methods": methods, "samplers": sampler_info, "prior_steps": args.prior_steps,
              "scope": "Fixed packaged LIBERO initial states, not certified unseen training geometry.", "records": []}
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        for field in ("protocol_id", "methods", "samplers", "prior_steps"):
            if previous[field] != report[field]:
                raise ValueError(f"Cannot resume changed {field}")
        report = previous
        if any(r["status"] != "complete" for r in report["records"]):
            raise ValueError("Prior errors retained; use a fresh output rather than silently overwriting failed cases")
        report["status"] = "running"
    write_json(result_path, report)
    completed = {case_key(r) for r in report["records"]}
    if len(completed) != len(report["records"]):
        raise ValueError("Duplicate saved rollout cases")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    try:
        bounds = tuple(np.asarray(x, np.float32) for x in env.env.action_spec)
        if any(x.shape != (7,) for x in bounds):
            raise ValueError("LIBERO action_spec is not 7-D")
        for scheduled in schedule:
            for method in methods:
                record = dict(scheduled, method=method, protocol_id=protocol_id)
                if case_key(record) in completed:
                    continue
                started = time.monotonic()
                try:
                    # Reset environment RNG and simulator state identically for each method.
                    np.random.seed(scheduled["env_seed"])
                    env.seed(scheduled["env_seed"])
                    env.reset()
                    obs = env.set_init_state(initial_states[scheduled["initial_state_index"]])
                    history = collections.deque([raw_frame(obs, 0)], maxlen=2)
                    done = False
                    for wait in range(args.wait_steps):
                        obs, _, wait_done, _ = env.step([0.] * 6 + [-1.])
                        history.append(raw_frame(obs, wait + 1))
                        if wait_done or env.check_success():
                            raise RuntimeError("Initial state succeeds during settling; audit before comparison")
                    plan = collections.deque()
                    executed, queries, clipped, out_of_bounds, total_coordinates = 0, 0, 0, 0, 0
                    retrieval = []
                    inference_seconds = []
                    while executed < args.max_steps and not done:
                        if not plan:
                            tick = time.monotonic()
                            images, state = policy_inputs(history, normalizer["state"])
                            with torch.inference_mode():
                                condition = base.condition(torch.from_numpy(images).to(args.device),
                                                           torch.from_numpy(state).to(args.device))
                                generator = torch.Generator(device=args.device).manual_seed(paired_noise_seed(
                                    args.task_name, scheduled["initial_state_index"], scheduled["policy_seed"], queries))
                                noise = torch.randn((1, 16, 7), generator=generator, device=args.device)
                                if method == "gaussian":
                                    source = noise
                                elif method == "zero":
                                    source = torch.zeros_like(noise)
                                elif method in ("train_nn", "train_cond_nn"):
                                    condition_cpu = condition.cpu().numpy()
                                    if method == "train_nn":
                                        key = visual_keys(condition_cpu)
                                        index = int(np.argmax(bank_keys @ key[0]))
                                    else:
                                        key = (condition_cpu[0] - retrieval_mean) / retrieval_std
                                        index = int(np.argmin(np.mean((full_bank_keys - key)**2, axis=1)))
                                    source = torch.from_numpy(train_source[index:index+1]).to(args.device)
                                    retrieval.append({"bank_index": index, "episode": str(train_episode[index])})
                                else:
                                    source = samplers[method].sample(condition, noise=noise, steps=args.prior_steps)
                                chunk = integrate(base, condition, source, args.decoder_steps, solver=args.decoder)[0].cpu().numpy()
                            inference_seconds.append(time.monotonic() - tick)
                            raw_actions, _ = executable_actions(chunk, normalizer["actions"])
                            out_of_bounds += int(np.count_nonzero((raw_actions < bounds[0]) | (raw_actions > bounds[1])))
                            actions, stats = executable_actions(chunk, normalizer["actions"], bounds if args.clip_env_actions else None)
                            clipped += stats["clipped_coordinates"]
                            total_coordinates += stats["coordinates"]
                            plan.extend(actions)
                            queries += 1
                        obs, _, done, _ = env.step(np.asarray(plan.popleft(), np.float32).tolist())
                        if bool(done) != bool(env.check_success()):
                            raise RuntimeError("Environment termination and explicit success predicate disagree")
                        executed += 1
                        history.append(raw_frame(obs, args.wait_steps + executed))
                    record.update(status="complete", success=bool(done), executed_steps=executed,
                                  policy_queries=queries, selected_train_rows=retrieval,
                                  inferred_coordinates=total_coordinates, clipped_coordinates=clipped,
                                  out_of_env_bounds_coordinates=out_of_bounds,
                                  mean_query_seconds=float(np.mean(inference_seconds)),
                                  elapsed_seconds=time.monotonic() - started)
                    report["records"].append(record)
                    completed.add(case_key(record))
                    write_json(result_path, report)
                    print(json.dumps({"method": method, "state": scheduled["initial_state_index"],
                                      "policy_seed": scheduled["policy_seed"], "success": bool(done),
                                      "completed": len(completed), "total": len(schedule)*len(methods)}), flush=True)
                except Exception:
                    record.update(status="error", error=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
                    report["records"].append(record)
                    report["status"] = "failed"
                    write_json(result_path, report)
                    raise
        if digest_file(args.action_checkpoint) != action_hash:
            raise RuntimeError("Action checkpoint file changed during evaluation")
        if len(report["records"]) != len(schedule) * len(methods):
            raise RuntimeError("Incomplete evaluation schedule")
        if len(methods) >= 2:
            report["paired_summary"] = paired_success_summary(report["records"], methods)
        report["status"] = "complete"
        write_json(result_path, report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
