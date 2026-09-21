"""Run PickCube with a learned expert-inversion Prior Flow.

The checkpoint loader, frozen EMA Action Flow, observation preprocessing,
normalizer, simulator loop, and success accounting all remain the standard
MomentVLA evaluation path.  Only action-chunk sampling is replaced by:

    frozen encoder -> c
    Gaussian -> 16-step Prior Flow -> z_prior
    z_prior -> native 10-step frozen Action Flow -> action

Use ``--method fm`` to run the untouched Gaussian Action Flow baseline through
the same evaluation entry point and with the same environment seed.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from roboverse_learn.il.policies.fm.latent_prediction.prior_flow_inference import (
    encode_frozen_baseline_condition,
    load_prior_flow,
    midpoint_action_flow,
    predict_action_with_prior,
)
from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner
from roboverse_learn.il.runners.default_runner import DefaultRunner


RUNNER_CONFIG: dict[str, Any] | None = None


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_integer_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(set(result)) != len(result):
        raise ValueError(f"values must be unique, got {result}")
    return result


def build_fixed_initial_state_trajectory(
    source_path: str | Path,
    output_dir: str | Path,
    robot: str,
    initial_episodes: list[int],
    source_seeds: list[int],
) -> tuple[Path, list[dict[str, int]]]:
    """Repeat selected task initial states once for every source seed."""
    source_path = Path(source_path).expanduser().resolve()
    with gzip.open(source_path, "rb") as handle:
        payload = pickle.load(handle)
    if robot not in payload:
        raise KeyError(f"robot {robot!r} is absent from {source_path}")
    trajectories = payload[robot]
    invalid = [index for index in initial_episodes if not 0 <= index < len(trajectories)]
    if invalid:
        raise IndexError(f"initial episode indices out of range: {invalid}")

    rows = []
    repeated = []
    for initial_episode in initial_episodes:
        for source_seed in source_seeds:
            trial_index = len(rows)
            repeated.append(copy.deepcopy(trajectories[initial_episode]))
            rows.append(
                {
                    "trial_index": trial_index,
                    "initial_episode": int(initial_episode),
                    "source_seed": int(source_seed),
                }
            )

    output_dir = Path(output_dir).expanduser().resolve() / "fixed_initial_states"
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_tag = "_".join(str(value) for value in initial_episodes)
    # MetaSim selects its trajectory parser from the literal "v2" marker in
    # the path, so retain that marker for this derived v2 trajectory.
    output_path = output_dir / (
        f"pick_cube_episodes_{episode_tag}_seeds_{source_seeds[0]}-"
        f"{source_seeds[-1]}_v2.pkl.gz"
    )
    repeated_payload = copy.deepcopy(payload)
    repeated_payload[robot] = repeated
    with gzip.open(output_path, "wb") as handle:
        pickle.dump(repeated_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(
        output_path.with_suffix(".json"),
        {
            "source_trajectory": str(source_path),
            "output_trajectory": str(output_path),
            "robot": robot,
            "initial_episodes": initial_episodes,
            "source_seeds": source_seeds,
            "num_trials": len(rows),
            "ordering": "initial_episode major, source_seed minor",
            "trials": rows,
        },
    )
    return output_path, rows


class SeededSourceMixin:
    """Generate auditable per-trial Gaussian sources for paired methods."""

    def _initialize_source_randomness(self, config: dict[str, Any]) -> None:
        self.noise_seed = int(config["noise_seed"])
        self._source_seed_schedule = [
            int(value) for value in config.get("source_seed_schedule", [])
        ]
        self._batch_index = -1
        self._query_index = 0
        self._source_generators: list[torch.Generator] = []
        self._current_source_seeds: list[int] = []

    def _reset_source_randomness(self) -> None:
        self._batch_index += 1
        self._query_index = 0
        start = self._batch_index * self.num_envs
        if self._source_seed_schedule:
            seeds = self._source_seed_schedule[start : start + self.num_envs]
            if len(seeds) != self.num_envs:
                raise RuntimeError(
                    f"source seed schedule ended at trial {start}; "
                    f"needed {self.num_envs}, found {len(seeds)}"
                )
        else:
            seeds = [self.noise_seed + start + index for index in range(self.num_envs)]
        self._current_source_seeds = seeds
        self._source_generators = []
        for seed in seeds:
            generator = torch.Generator(device=torch.device(self.device))
            generator.manual_seed(seed)
            self._source_generators.append(generator)

    def _sample_source(self, dtype: torch.dtype) -> torch.Tensor:
        pieces = [
            torch.randn(
                (1, int(self.policy.horizon), int(self.policy.action_dim)),
                device=self.device,
                dtype=dtype,
                generator=generator,
            )
            for generator in self._source_generators
        ]
        return torch.cat(pieces, dim=0)

    def _append_trace(self, value: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, allow_nan=False) + "\n")


class PriorFlowEvalRunner(SeededSourceMixin, DefaultEvalRunner):
    """Standard evaluator whose action source is the learned Prior Flow."""

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        super()._init_policy(default_runner, **kwargs)
        config = RUNNER_CONFIG
        if config is None:
            raise RuntimeError("RUNNER_CONFIG was not initialized")

        # DefaultEvalRunner.load_payload restores the checkpoint's historical
        # output directory, so redirect it back to this evaluation run.
        default_runner._output_dir = config["run_dir"]
        self.policy.eval().requires_grad_(False)
        self.prior_flow, payload = load_prior_flow(
            config["prior_checkpoint"], self.device
        )

        expected_hash = payload.get("action_flow_checkpoint_sha256")
        if (
            expected_hash != config["action_flow_checkpoint_sha256"]
            and not config["allow_action_checkpoint_mismatch"]
        ):
            raise ValueError(
                "Prior Flow was trained for a different Action Flow checkpoint: "
                f"prior={expected_hash}, rollout={config['action_flow_checkpoint_sha256']}"
            )
        latent_shape = tuple(int(value) for value in payload["model_config"]["latent_shape"])
        expected_shape = (int(self.policy.horizon), int(self.policy.action_dim))
        if latent_shape != expected_shape:
            raise ValueError(
                f"Prior latent shape {latent_shape} does not match policy {expected_shape}"
            )

        self.prior_payload = payload
        self.prior_steps = int(config["prior_steps"])
        self.prior_depth = float(config["prior_depth"])
        self.action_steps = int(config["action_steps"])
        self.trace_path = Path(config["trace_path"])
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_source_randomness(config)

        audit = {
            "event": "initialized",
            "prior_checkpoint": config["prior_checkpoint"],
            "prior_checkpoint_sha256": config["prior_checkpoint_sha256"],
            "prior_epoch": int(payload["epoch"]),
            "prior_global_step": int(payload["global_step"]),
            "prior_validation_loss": float(payload["validation_loss"]),
            "action_flow_checkpoint_sha256": expected_hash,
            "latent_shape": list(latent_shape),
            "condition_dim": int(self.prior_flow.condition_dim),
            "prior_steps": self.prior_steps,
            "prior_depth": self.prior_depth,
            "prior_steps_executed": int(self.prior_depth * self.prior_steps + 1e-12),
            "action_steps": self.action_steps,
            "source_seed_schedule": self._source_seed_schedule,
            "action_flow_frozen": not any(
                parameter.requires_grad for parameter in self.policy.parameters()
            ),
            "prior_flow_frozen": not any(
                parameter.requires_grad for parameter in self.prior_flow.parameters()
            ),
        }
        self._append_trace(audit)

    def reset(self):
        super().reset()
        self._reset_source_randomness()

    @torch.inference_mode()
    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()
        source = self._sample_source(dtype=next(self.policy.parameters()).dtype)
        started = time.perf_counter()
        result = predict_action_with_prior(
            self.policy,
            self.prior_flow,
            obs,
            prior_steps=self.prior_steps,
            prior_depth=self.prior_depth,
            action_steps=self.action_steps,
            noise=source,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._append_trace(
            {
                "event": "action_query",
                "batch_index": self._batch_index,
                "query_index": self._query_index,
                "env_step": int(self.step),
                "batch_size": int(result["action"].shape[0]),
                "source_seeds": self._current_source_seeds,
                "source_mean": float(source.mean()),
                "source_std": float(source.std(unbiased=False)),
                "source_sum": float(source.sum()),
                "source_square_sum": float(source.square().sum()),
                "condition_shape": list(result["condition"].shape),
                "z_prior_shape": list(result["z_prior"].shape),
                "z_prior_mean": float(result["z_prior"].mean()),
                "z_prior_std": float(result["z_prior"].std(unbiased=False)),
                "z_prior_rms": float(result["z_prior"].square().mean().sqrt()),
                "action_min": float(result["action"].min()),
                "action_max": float(result["action"].max()),
                "elapsed_ms": elapsed_ms,
            }
        )
        self._query_index += 1
        return result["action"].detach().to(torch.float32).transpose(0, 1)


class PriorFlowDefaultRunner(DefaultRunner):
    @staticmethod
    def get_eval_runner_class():
        return PriorFlowEvalRunner


class FMBaselineEvalRunner(SeededSourceMixin, DefaultEvalRunner):
    """Gaussian FM baseline with an explicit, auditable source tensor."""

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        super()._init_policy(default_runner, **kwargs)
        config = RUNNER_CONFIG
        if config is None:
            raise RuntimeError("RUNNER_CONFIG was not initialized")
        default_runner._output_dir = config["run_dir"]
        self.policy.eval().requires_grad_(False)
        self.action_steps = int(config["action_steps"])
        self.trace_path = Path(config["trace_path"])
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_source_randomness(config)
        self._append_trace(
            {
                "event": "initialized",
                "method": "fm",
                "action_flow_checkpoint_sha256": config[
                    "action_flow_checkpoint_sha256"
                ],
                "latent_shape": [
                    int(self.policy.horizon),
                    int(self.policy.action_dim),
                ],
                "action_steps": self.action_steps,
                "source_seed_schedule": self._source_seed_schedule,
                "action_flow_frozen": not any(
                    parameter.requires_grad for parameter in self.policy.parameters()
                ),
            }
        )

    def reset(self):
        super().reset()
        self._reset_source_randomness()

    @torch.inference_mode()
    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()
        condition = encode_frozen_baseline_condition(self.policy, obs)
        source = self._sample_source(dtype=condition.dtype)
        started = time.perf_counter()
        normalized_action = midpoint_action_flow(
            self.policy, source, condition, steps=self.action_steps
        )
        action_pred = self.policy.normalizer["action"].unnormalize(normalized_action)
        start = self.policy.n_obs_steps - 1
        end = start + self.policy.n_action_steps
        action = action_pred[:, start:end]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._append_trace(
            {
                "event": "action_query",
                "batch_index": self._batch_index,
                "query_index": self._query_index,
                "env_step": int(self.step),
                "batch_size": int(action.shape[0]),
                "source_seeds": self._current_source_seeds,
                "source_mean": float(source.mean()),
                "source_std": float(source.std(unbiased=False)),
                "source_sum": float(source.sum()),
                "source_square_sum": float(source.square().sum()),
                "condition_shape": list(condition.shape),
                "action_min": float(action.min()),
                "action_max": float(action.max()),
                "elapsed_ms": elapsed_ms,
            }
        )
        self._query_index += 1
        return action.detach().to(torch.float32).transpose(0, 1)


class FMBaselineDefaultRunner(DefaultRunner):
    @staticmethod
    def get_eval_runner_class():
        return FMBaselineEvalRunner


def parse_final_stats(run_dir: Path) -> dict[str, Any]:
    candidates = sorted(
        run_dir.rglob("final_stats.txt"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        return {"status": "missing_final_stats", "run_dir": str(run_dir)}
    path = candidates[-1]
    text = path.read_text(errors="replace")
    success = re.search(r"Total Success:\s*(\d+)", text)
    trials = re.search(r"Total Completed:\s*(\d+)", text)
    rate = re.search(r"Average Success Rate:\s*([0-9.]+)", text)
    latency = re.search(r"Average Inference Time:\s*([0-9.]+)ms", text)
    return {
        "status": "completed",
        "final_stats": str(path),
        "successes": int(success.group(1)) if success else None,
        "trials": int(trials.group(1)) if trials else None,
        "success_rate": float(rate.group(1)) if rate else None,
        "average_inference_ms": float(latency.group(1)) if latency else None,
    }


def run_rollout(args: argparse.Namespace) -> dict[str, Any]:
    global RUNNER_CONFIG
    if not 0.0 <= float(args.prior_depth) <= 1.0:
        raise ValueError(f"--prior-depth must be in [0, 1], got {args.prior_depth}")
    action_checkpoint = Path(args.action_checkpoint).expanduser().resolve()
    prior_checkpoint = Path(args.prior_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_name = args.run_name or args.method
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "prior_flow_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    fixed_initial_episodes = parse_integer_list(args.fixed_initial_episodes)
    source_seeds = list(
        range(
            int(args.source_seed_start),
            int(args.source_seed_start + args.source_seed_count),
        )
    )
    sweep_trajectory = None
    sweep_trials: list[dict[str, int]] = []
    if fixed_initial_episodes:
        if not source_seeds:
            raise ValueError("--source-seed-count must be positive for a fixed-state sweep")
        sweep_trajectory, sweep_trials = build_fixed_initial_state_trajectory(
            source_path=args.eval_traj_path,
            output_dir=output_dir,
            robot=args.robot,
            initial_episodes=fixed_initial_episodes,
            source_seeds=source_seeds,
        )

    action_hash = sha256(action_checkpoint)
    prior_hash = sha256(prior_checkpoint)
    RUNNER_CONFIG = {
        "method": args.method,
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "action_flow_checkpoint_sha256": action_hash,
        "prior_checkpoint": str(prior_checkpoint),
        "prior_checkpoint_sha256": prior_hash,
        "prior_steps": int(args.prior_steps),
        "prior_depth": float(args.prior_depth),
        "action_steps": int(args.action_steps),
        "allow_action_checkpoint_mismatch": bool(
            args.allow_action_checkpoint_mismatch
        ),
        "noise_seed": int(args.noise_seed),
        "source_seed_schedule": [row["source_seed"] for row in sweep_trials],
    }

    payload = torch.load(
        action_checkpoint.open("rb"),
        pickle_module=__import__("dill"),
        map_location="cpu",
    )
    runner_class = (
        PriorFlowDefaultRunner if args.method == "prior" else FMBaselineDefaultRunner
    )
    workspace = runner_class(payload["cfg"], output_dir=str(run_dir))
    workspace.cfg.dataset_config.zarr_path = str(
        Path(args.zarr_path).expanduser().resolve()
    )
    eval_args = workspace.eval_args
    eval_args.task = args.task
    eval_args.robot = args.robot
    eval_args.sim = args.sim
    eval_args.num_envs = int(args.num_envs)
    num_trials = len(sweep_trials) if sweep_trials else int(args.max_demos)
    start_demo = 0 if sweep_trials else int(args.start_demo)
    eval_args.max_demo = num_trials
    eval_args.task_id_range_low = start_demo
    eval_args.task_id_range_high = start_demo + num_trials
    eval_args.max_step = int(args.max_steps)
    eval_args.headless = True
    eval_args.gpu_id = int(args.gpu_id)
    eval_args.level = int(args.level)
    eval_args.scene_mode = int(args.scene_mode)
    eval_args.randomization_seed = int(args.environment_seed)
    eval_args.cube_shift_x = 0.0
    eval_args.cube_shift_y = 0.0
    eval_args.cube_shift_object = "cube"
    eval_args.cube_shift_step = -1
    eval_args.save_video_freq = int(args.save_video_freq)
    eval_args.subset = f"prior_flow_{args.method}"

    run_config = {
        **RUNNER_CONFIG,
        "action_checkpoint": str(action_checkpoint),
        "zarr_path": workspace.cfg.dataset_config.zarr_path,
        "task": args.task,
        "robot": args.robot,
        "sim": args.sim,
        "num_envs": int(args.num_envs),
        "start_demo": start_demo,
        "max_demos": num_trials,
        "max_steps": int(args.max_steps),
        "level": int(args.level),
        "scene_mode": int(args.scene_mode),
        "environment_seed": int(args.environment_seed),
        "fixed_initial_episodes": fixed_initial_episodes,
        "source_seeds": source_seeds if fixed_initial_episodes else None,
        "sweep_trajectory": str(sweep_trajectory) if sweep_trajectory else None,
        "sweep_trials": sweep_trials,
        "paired_source_contract": (
            "same source_seed and exact explicit Gaussian tensor per trial/query"
            if fixed_initial_episodes
            else "deterministic per-trial source generator"
        ),
        "composition": (
            "Gaussian -> Prior Flow -> frozen Action Flow"
            if args.method == "prior"
            else "Gaussian -> frozen Action Flow"
        ),
    }
    write_json(run_dir / "rollout_config.json", run_config)

    task_cls = None
    original_trajectory_path = None
    if sweep_trajectory is not None:
        from metasim.task.registry import get_task_class

        task_cls = get_task_class(args.task)
        original_trajectory_path = task_cls.traj_filepath
        task_cls.traj_filepath = str(sweep_trajectory)

    started = time.perf_counter()
    try:
        workspace.evaluate(ckpt_path=action_checkpoint)
    finally:
        if task_cls is not None:
            task_cls.traj_filepath = original_trajectory_path
    result = parse_final_stats(run_dir)
    result.update(
        {
            "method": args.method,
            "elapsed_seconds": time.perf_counter() - started,
            "run_dir": str(run_dir),
            "trace_path": str(trace_path) if args.method == "prior" else None,
        }
    )
    write_json(run_dir / "rollout_result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("prior", "fm"), default="prior")
    parser.add_argument(
        "--run-name",
        default="",
        help="Output subdirectory name; defaults to the method name.",
    )
    parser.add_argument(
        "--action-checkpoint",
        default="/data/yiming/MomentVLA-main/il_outputs/fm_unet/pick_cube_isaac100/checkpoints/30.ckpt",
    )
    parser.add_argument(
        "--prior-checkpoint",
        default="/data/yiming/MomentVLA-main/il_outputs/prior_flow/pick_cube_expert_inverse_unet_same_as_actionflow_20260912/best.pt",
    )
    parser.add_argument(
        "--zarr-path",
        default="/data/yiming/MomentVLA-main/data_policy/pick_cubeIsaacSimL0_obs:joint_pos_act:joint_pos_100.zarr",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yiming/MomentVLA-main/il_outputs/prior_flow/pick_cube_expert_inverse_unet_same_as_actionflow_20260912/rollouts",
    )
    parser.add_argument("--task", default="pick_cube")
    parser.add_argument("--robot", default="franka")
    parser.add_argument("--sim", default="isaacsim")
    parser.add_argument("--prior-steps", type=int, default=16)
    parser.add_argument("--prior-depth", type=float, default=1.0)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument(
        "--allow-action-checkpoint-mismatch",
        action="store_true",
        help="Allow evaluating a Prior trained against another compatible Action Flow.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--start-demo", type=int, default=0)
    parser.add_argument("--max-demos", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--level", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--scene-mode", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--environment-seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--save-video-freq", type=int, default=1_000_000)
    parser.add_argument(
        "--fixed-initial-episodes",
        default="",
        help="Comma-separated original trajectory indices to repeat for a seed sweep.",
    )
    parser.add_argument("--source-seed-start", type=int, default=0)
    parser.add_argument("--source-seed-count", type=int, default=20)
    parser.add_argument(
        "--eval-traj-path",
        default="/data/yiming/MomentVLA-main/roboverse_data/trajs/maniskill/pick_cube/v2/franka_v2.pkl.gz",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_rollout(args)
        return

    # IsaacSim may use os._exit after its timeout-guarded close. Keep result
    # aggregation in this lightweight parent so rollout_result.json is still
    # written even when the simulator worker takes that expected exit path.
    started = time.perf_counter()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    completed = subprocess.run(command, check=False)
    run_name = args.run_name or args.method
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    result = parse_final_stats(run_dir)
    result.update(
        {
            "method": args.method,
            "worker_returncode": int(completed.returncode),
            "elapsed_seconds": time.perf_counter() - started,
            "run_dir": str(run_dir),
            "trace_path": (
                str(run_dir / "prior_flow_trace.jsonl")
                if args.method == "prior"
                else None
            ),
        }
    )
    write_json(run_dir / "rollout_result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
