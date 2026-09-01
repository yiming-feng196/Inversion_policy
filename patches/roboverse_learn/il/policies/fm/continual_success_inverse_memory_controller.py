"""Cross-process controller for online, success-only inverse-memory growth.

Each IsaacSim evaluation is intentionally a separate child process because
DefaultEvalRunner force-closes Isaac at episode completion.  This controller
therefore provides the episode-boundary update without changing that runner:

    evaluate episode i -> read official SuccessOnce -> append only if success
    -> evaluate episode i+1 with the updated bank.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def build_randomized_tasks(source: Path, output: Path, seed: int, std_rad: float) -> Path:
    if output.exists():
        return output
    with gzip.open(source, "rb") as handle:
        payload = pickle.load(handle)
    randomized = copy.deepcopy(payload)
    rng = np.random.default_rng(seed)
    joints = [f"panda_joint{i}" for i in range(1, 8)]
    for trajectory in randomized["franka"]:
        dof_pos = trajectory["init_state"]["franka"]["dof_pos"]
        for name in joints:
            if name not in dof_pos:
                raise RuntimeError(f"Missing {name} in task trajectory")
            dof_pos[name] += float(rng.normal(0.0, std_rad))
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as handle:
        pickle.dump(randomized, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output.with_suffix(".json"), {
        "source": str(source), "output": str(output), "init_seed": seed,
        "init_joint_std_rad": std_rad, "num_episodes": len(randomized["franka"]),
    })
    return output


def parse_success(episode_dir: Path, task_index: int) -> bool:
    candidates = list(episode_dir.rglob(f"{task_index:04d}.txt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one outcome file for task {task_index}, found {len(candidates)}")
    matched = re.search(r"SuccessOnce:\s*(True|False)", candidates[0].read_text(errors="replace"))
    if matched is None:
        raise RuntimeError(f"No SuccessOnce in {candidates[0]}")
    return matched.group(1) == "True"


def run_child(args: argparse.Namespace, stream: str, task_index: int, memory: Path,
              capture: bool, output_dir: Path) -> tuple[bool, float]:
    episode_dir = output_dir / "episodes" / f"episode_{task_index:03d}"
    capture_dir = output_dir / "capture" if capture else None
    command = [
        args.isaac_python, args.core_script,
        "--stage", "rollout",
        "--zarr-path", args.zarr_path,
        "--demo-root", args.demo_root,
        "--checkpoint", args.checkpoint,
        "--memory-path", str(memory),
        "--output-dir", str(episode_dir),
        "--task", args.task, "--robot", args.robot, "--sim", args.sim,
        "--gpu-id", "0", "--device", "cuda:0",
        "--max-demos", "1", "--num-episodes", "1",
        "--task-id-start", str(task_index),
        "--max-steps", str(args.max_steps),
        # The policy runner validates the requested source depth even for the
        # Gaussian baseline.  It must therefore match the depth actually
        # stored in the selected memory, rather than StackCube's old x0 value.
        "--depth", str(args.source_depth), "--flow-steps", str(args.flow_steps),
        "--inverse-steps", "8", "--horizon", "16", "--n-obs-steps", "8",
        "--methods", "vanilla_gaussian_flow" if stream == "gaussian" else "flow_condition_qkv_inverse",
        "--shifts", "0", "--rollout-depths", str(args.source_depth),
        "--seed", str(args.policy_seed + task_index),
        "--qkv-top-m", "4", "--qkv-value-mode", args.qkv_value_mode,
        "--qkv-temporal-history", str(args.qkv_temporal_history),
        "--qkv-spatial-pool", str(args.qkv_spatial_pool),
        "--save-video-freq", "1000000",
    ]
    if args.qkv_sequence_lock:
        command.append("--qkv-sequence-lock")
    if args.qkv_executed_window_compatibility:
        command.append("--qkv-executed-window-compatibility")
    if args.qkv_force_validated_successor:
        command.append("--qkv-force-validated-successor")
    if args.qkv_validated_escape_lock:
        command += [
            "--qkv-validated-escape-lock",
            "--qkv-escape-semantic-tolerance",
            str(args.qkv_escape_semantic_tolerance),
        ]
    if capture:
        assert capture_dir is not None
        capture_dir.mkdir(parents=True, exist_ok=True)
        command += ["--capture-source-records", "--capture-source-dir", str(capture_dir)]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "1",
        "TMPDIR": "/data/yiming/tmp_codex",
        "XDG_CACHE_HOME": "/data/yiming/cache_codex",
        "CUDA_CACHE_PATH": "/data/yiming/cuda_cache",
        "OPTIX_CACHE_PATH": "/data/yiming/optix_cache",
    })
    episode_dir.mkdir(parents=True, exist_ok=True)
    log_path = episode_dir / "run.log"
    import time
    started = time.perf_counter()
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    elapsed = time.perf_counter() - started
    # Isaac's timeout-guarded close exits its child process after results are
    # written.  The outcome file is the authoritative status, not returncode.
    try:
        success = parse_success(episode_dir, task_index)
    except Exception:
        tail = log_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(f"Episode {task_index} produced no valid result (return={completed.returncode}):\n{tail}")
    return success, elapsed


def append_records(args: argparse.Namespace, memory: Path, capture_dir: Path,
                   task_index: int, output_dir: Path) -> dict[str, Any]:
    summary = output_dir / "insertions" / f"episode_{task_index:03d}.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.isaac_python, args.append_script,
        "--memory-path", str(memory), "--capture-dir", str(capture_dir),
        "--episode-id", str(task_index), "--summary-path", str(summary),
    ]
    environment = os.environ.copy()
    environment["TMPDIR"] = "/data/yiming/tmp_codex"
    completed = subprocess.run(command, text=True, capture_output=True, env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"Memory append failed: {completed.stderr}\n{completed.stdout}")
    return json.loads(summary.read_text())


def run_stream(args: argparse.Namespace, stream: str, randomized_path: Path) -> dict[str, Any]:
    root = Path(args.output_dir).resolve() / stream
    root.mkdir(parents=True, exist_ok=True)
    working = root / "working_memory.pt"
    if stream != "gaussian":
        shutil.copy2(args.base_memory_path, working)
    rows = []
    for task_index in range(args.num_episodes):
        success, elapsed = run_child(
            args, stream, task_index,
            Path(args.base_memory_path) if stream in {"gaussian", "static"} else working,
            capture=stream == "continual", output_dir=root,
        )
        insertion: dict[str, Any] | None = None
        if stream == "continual" and success:
            insertion = append_records(args, working, root / "capture", task_index, root)
        row = {
            "task_index": task_index, "success": success,
            "elapsed_seconds": elapsed,
            "inserted_records": 0 if insertion is None else insertion["inserted_records"],
            "bank_items_after": None if stream == "gaussian" else (
                None if insertion is None else insertion["bank_items_after"]
            ),
        }
        rows.append(row)
        print(json.dumps({"stream": stream, **row}), flush=True)
    result = {
        "stream": stream,
        "num_episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / max(1, len(rows)),
        "temporal_causality": "episode i reads only base memory plus entries from successes in episodes < i",
        "randomized_trajectory": str(randomized_path),
        "rows": rows,
    }
    write_json(root / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--core-script", required=True)
    parser.add_argument("--append-script", required=True)
    parser.add_argument("--base-memory-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zarr-path", required=True)
    parser.add_argument("--demo-root", required=True)
    parser.add_argument("--eval-traj-path", required=True)
    parser.add_argument("--output-dir", required=True)
    # Keep the cross-process continual-memory machinery task-agnostic.  The
    # runner continues to own task construction; we only select which existing
    # registered task and trajectory file a child process evaluates.
    parser.add_argument("--task", default="stack_cube")
    parser.add_argument("--robot", default="franka")
    parser.add_argument("--sim", default="isaacsim")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument(
        "--source-depth", type=float, default=0.8,
        help="Flow depth stored by the selected inverse-memory payload.",
    )
    parser.add_argument(
        "--qkv-value-mode",
        default="compatibility",
        choices=[
            "compatibility", "temporal_compatibility",
            "geometric_temporal_compatibility",
        ],
    )
    parser.add_argument("--qkv-temporal-history", type=int, default=3)
    parser.add_argument("--qkv-spatial-pool", type=int, default=128)
    parser.add_argument(
        "--qkv-sequence-lock", action="store_true",
        help="Advance retrieval only along the expert path selected for the executed chunk.",
    )
    parser.add_argument(
        "--qkv-executed-window-compatibility", action="store_true",
        help="Select inverse sources using only DefaultEvalRunner's executed action slice.",
    )
    parser.add_argument(
        "--qkv-force-validated-successor", action="store_true",
        help="Force a validated prior-path successor; use global retrieval only after rejection.",
    )
    parser.add_argument(
        "--qkv-validated-escape-lock",
        action="store_true",
        help=(
            "Keep global geometric-temporal retrieval available and add a prior "
            "trajectory continuation only when the live state validates it."
        ),
    )
    parser.add_argument(
        "--qkv-escape-semantic-tolerance", type=float, default=0.01,
        help="Relative Flow-condition cosine tolerance used by validated escape.",
    )
    parser.add_argument("--policy-seed", type=int, default=4242)
    parser.add_argument("--init-seed", type=int, default=20260901)
    parser.add_argument("--init-qpos-std", type=float, default=0.05)
    parser.add_argument(
        "--randomized-traj-path",
        default=None,
        help="Reuse an existing randomized trajectory file for exact paired streams.",
    )
    parser.add_argument("--streams", default="static,continual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    randomized_path = (
        Path(args.randomized_traj_path).resolve()
        if args.randomized_traj_path else build_randomized_tasks(
            Path(args.eval_traj_path).resolve(),
            root / "randomized_initial_states" /
            f"{args.task}_seed{args.init_seed}_q{args.init_qpos_std:g}.pkl.gz",
            args.init_seed, args.init_qpos_std,
        )
    )
    if not randomized_path.exists():
        raise FileNotFoundError(randomized_path)
    # The task registry path is process-local, so each child needs to see the
    # randomized trajectory via a tiny path override file read by the core.
    # The original evaluator already honors this environment variable in the
    # patch below; keeping it external avoids edits to DefaultEvalRunner.
    os.environ["EVAL_TRAJ_PATH_OVERRIDE"] = str(randomized_path)
    results = {}
    for stream in [value.strip() for value in args.streams.split(",") if value.strip()]:
        results[stream] = run_stream(args, stream, randomized_path)
    write_json(root / "summary.json", {
        "experiment": "continual_success_only_inverse_memory",
        "task": args.task,
        "robot": args.robot,
        "sim": args.sim,
        "initial_joint_std_rad": args.init_qpos_std,
        "source_depth": args.source_depth,
        "results": results,
    })


if __name__ == "__main__":
    main()
