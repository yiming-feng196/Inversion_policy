"""Deployable native-IsaacSim rollout for Gaussian vs temporal reuse.

The first query uses a Gaussian source for both methods.  After a query has
produced a full raw action chunk, ``previous_region_reuse`` inverts that
chunk under the previous causal condition and directly reuses the resulting
source at the next query.  No expert action, validation latent, oracle choice,
gate, or correction network is read by this runner.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


def _get(config, key, default):
    return config.get(key, default)


def _obs_dict(obs, robot_name):
    return {
        "rgb": obs.cameras["camera0"].rgb,
        "joint_qpos": obs.robots[robot_name].joint_pos,
    }


@torch.no_grad()
def run(config: dict):
    repo = str(_get(config, "repo", "/home/yiming/MomentVLA-main"))
    code_dir = str(Path(__file__).resolve().parent)
    for item in (code_dir, repo):
        if item not in sys.path:
            sys.path.insert(0, item)

    from flow_latent_predictor_common import (
        forward_flow,
        load_flow,
        prepare_predictor_context,
        reverse_flow,
        seed_all,
    )

    seed = int(_get(config, "seed", 0))
    seed_all(seed)
    device = str(_get(config, "device", "cuda:0"))
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
    args = type("Args", (), {})()
    args.repo = repo
    args.checkpoint = str(_get(config, "checkpoint", "/data/yiming/MomentVLA-main/il_outputs/fm_unet/pick_cube_isaac100/checkpoints/30.ckpt"))
    args.device = device
    policy, matcher, flow_cfg = load_flow(args)
    nfe = int(_get(config, "nfe", 200))
    if nfe != 200:
        raise ValueError("formal rollout uses 200 Flow steps")

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
    from metasim.task.registry import get_task_class
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot
    from roboverse_pack.tasks.maniskill import pick_cube as _pick_cube_task  # noqa: F401
    from roboverse_learn.il.configs.base_config import DiffusionPolicyCfg
    from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner

    task_name = str(_get(config, "task", "pick_cube"))
    robot_name = str(_get(config, "robot", "franka"))
    method = str(_get(config, "method", "gaussian"))
    if method not in {"gaussian", "previous_region_reuse"}:
        raise ValueError(f"unsupported deployable method: {method}")
    task_cls = get_task_class(task_name)
    camera = PinholeCameraCfg(name="camera0", data_types=["rgb", "depth"], width=256, height=256,
                              pos=(1.0, 0.0, 0.75), look_at=(0.0, 0.0, 0.0))
    lights = [
        DiskLightCfg(name="ceiling_main", intensity=12000.0, color=(1.0, 1.0, 1.0), radius=1.2,
                     pos=(0.0, 0.0, 2.8), rot=(0.7071, 0.0, 0.0, 0.7071)),
        SphereLightCfg(name="ceiling_ne", intensity=5000.0, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)),
        SphereLightCfg(name="ceiling_nw", intensity=5000.0, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)),
        SphereLightCfg(name="ceiling_sw", intensity=5000.0, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)),
        SphereLightCfg(name="ceiling_se", intensity=5000.0, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)),
    ]
    scenario = task_cls.scenario.update(robots=[robot_name], simulator="isaacsim", num_envs=1,
                                        headless=True, lights=lights, cameras=[camera])
    os.environ["METASIM_FORCE_EXIT_ON_CLOSE"] = "1"
    os.environ.setdefault("METASIM_CLOSE_TIMEOUT_SEC", "8")
    env = task_cls(scenario, device=torch.device(device))
    initial_states, _, _ = get_traj(env.traj_filepath, get_robot(robot_name), env.handler)
    joint_names = sorted(scenario.robots[0].joint_limits)
    episodes = min(int(_get(config, "episodes", 1)), len(initial_states))
    shift_cm = float(str(_get(config, "scenario", "shift_0cm")).replace("shift_", "").replace("cm", ""))
    history_len = int(_get(config, "history", 12))
    max_steps = int(_get(config, "max_steps", 250))

    class ReuseRunner(DefaultEvalRunner):
        def __init__(self, *runner_args, **kwargs):
            self.flow_policy = kwargs.pop("flow_policy")
            self.flow_cfg = kwargs.pop("flow_cfg")
            self.matcher = kwargs.pop("matcher")
            self.history_len = kwargs.pop("history_len")
            self.flow_steps = kwargs.pop("flow_steps")
            self.method = kwargs.pop("method")
            self.obs = deque(maxlen=self.history_len)
            self.previous_chunk = None
            self.previous_condition = None
            self.current_full_chunk = None
            self.current_condition = None
            self.executed_buffer = []
            self.query_times_ms = []
            self.query_count = 0
            super().__init__(*runner_args, **kwargs)

        def _init_policy(self, default_runner, **kwargs):
            self.policy = self.flow_policy
            self.yaml_cfg = self.flow_cfg
            self.policy_cfg = DiffusionPolicyCfg()
            runner_cfg = self.flow_cfg.eval_config.policy_runner
            self.policy_cfg.obs_config.from_dict(runner_cfg.obs)
            self.policy_cfg.action_config.from_dict(runner_cfg.action)
            self.policy_cfg.obs_config.obs_dim = self.flow_cfg.shape_meta.obs.agent_pos.shape[0]
            self.policy_cfg.action_config.action_dim = self.flow_cfg.shape_meta.action.shape[0]
            self.policy_cfg.action_config.action_chunk_steps = self.policy.n_action_steps
            self.env = None

        def _get_history_obs(self):
            if not self.obs:
                raise RuntimeError("empty observation history")
            result = {}
            for key in self.obs[0]:
                values = [item[key] for item in self.obs]
                values = [values[0]] * (self.history_len - len(values)) + values
                result[key] = torch.stack(values[-self.history_len:], dim=1)
            return result

        def reset(self):
            super().reset()
            self.obs.clear()
            self.previous_chunk = None
            self.previous_condition = None
            self.current_full_chunk = None
            self.current_condition = None
            self.executed_buffer = []
            self.query_times_ms = []
            self.query_count = 0

        def predict_action(self, observation=None):
            if observation is not None:
                self.obs.append(observation)
            encoded = prepare_predictor_context(self.policy, self._get_history_obs())
            condition = encoded[:, -self.policy.n_obs_steps:].flatten(1)
            if self.method == "previous_region_reuse" and self.previous_chunk is not None:
                normalized = self.policy.normalizer["action"].normalize(self.previous_chunk)
                z = reverse_flow(self.policy, self.matcher, normalized, self.previous_condition, self.flow_steps)
            else:
                z = torch.randn((condition.shape[0], self.policy.horizon, self.policy.action_dim), device=condition.device)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            begin = time.perf_counter()
            normalized_action = forward_flow(self.policy, self.matcher, z, condition, self.flow_steps, recompute=False)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            self.query_times_ms.append((time.perf_counter() - begin) * 1000.0)
            raw = self.policy.normalizer["action"].unnormalize(normalized_action)
            self.current_full_chunk = raw.detach().clone()
            self.current_condition = condition.detach().clone()
            self.query_count += 1
            start = self.policy.n_obs_steps - 1
            end = start + self.policy.n_action_steps
            return raw[:, start:end].transpose(0, 1).to(torch.float32)

        def get_action(self, obs):
            # BaseEvalRunner queries a new chunk only when its internal action
            # cache is empty.  At that boundary the previous chunk has been
            # sent to the simulator one action at a time; fold those exact
            # targets back into the corresponding full Flow chunk before the
            # next predict_action call performs reverse integration.
            if len(self.action_cache) == 0 and self.current_full_chunk is not None and self.executed_buffer:
                executed = torch.as_tensor(self.executed_buffer, dtype=torch.float32, device=device).unsqueeze(0)
                start = self.policy.n_obs_steps - 1
                end = start + self.policy.n_action_steps
                if executed.shape[1] != end - start:
                    raise RuntimeError(f"executed action buffer has {executed.shape[1]} steps, expected {end-start}")
                self.previous_chunk = self.current_full_chunk.clone()
                self.previous_chunk[:, start:end] = executed
                self.previous_condition = self.current_condition.clone()
                self.executed_buffer = []
            action_dicts = super().get_action(obs)
            env_action = action_dicts[0]
            targets = env_action[self.scenario.robots[0].name]["dof_pos_target"]
            self.executed_buffer.append([targets[name] for name in joint_names])
            return action_dicts

    runner = ReuseRunner(None, scenario=scenario, num_envs=1, device=device, task_name=task_name,
                         flow_policy=policy, flow_cfg=flow_cfg, matcher=matcher,
                         history_len=history_len, flow_steps=nfe, method=method)
    rows = []
    try:
        for ep, initial_state in enumerate(initial_states[:episodes]):
            state = copy.deepcopy(initial_state)
            state["objects"]["cube"]["pos"][0] += shift_cm / 100.0
            obs, _ = env.reset(states=[state])
            runner.reset()
            initial_z = float(state["objects"]["cube"]["pos"][2])
            success_once = False
            grasped = False
            lift = False
            max_lift = 0.0
            timeout = False
            for step in range(max_steps):
                policy_obs = _obs_dict(obs, robot_name)
                actions = runner.get_action(policy_obs)
                obs, _, success, time_out, _ = env.step(actions)
                current_state = env.handler.get_states()
                cube_pos = current_state.objects["cube"].root_state[0, :3]
                cube_z = float(cube_pos[2].detach().cpu())
                lift_height = cube_z - initial_z
                max_lift = max(max_lift, lift_height)
                grasped |= lift_height >= 0.01
                lift |= lift_height >= 0.10
                success_once |= bool(success[0])
                timeout |= bool(time_out[0])
                if success_once or timeout:
                    break
            rows.append({
                "episode": ep,
                "success": bool(success_once),
                "grasped_once": bool(grasped),
                "lift_success": bool(lift),
                "episode_length": step + 1,
                "max_cube_lift_m": max_lift,
                "failure_type": "success" if success_once else ("timeout_after_grasp" if grasped and timeout else "grasped_but_not_lifted" if grasped else "no_grasp"),
                "policy_queries": runner.query_count,
                "mean_inference_ms": float(np.mean(runner.query_times_ms)) if runner.query_times_ms else 0.0,
                "method_definition": "initial Gaussian, then invert previous executed action slice embedded in the full chunk and reuse",
            })
    finally:
        # IsaacSim 5 can hang while closing a headless RTX app.  The rollout
        # harness runs each method in a fresh process, so leave cleanup to the
        # process boundary after the CSV writer has received these rows.
        if bool(_get(config, "close_env", False)):
            env.close()
    return rows
