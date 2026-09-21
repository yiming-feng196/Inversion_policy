"""Small native-IsaacSim runner for the temporal-region staleness smoke test.

The runner replays the recorded expert action sequence after shifting the cube
at reset.  It computes the decoder error of the current inversion and of the
previous query's inverted action chunk under the current observation.  This
is deliberately labelled as a *replayed demo action target* diagnostic: the
repository does not currently contain an expert controller that replans
``A_t^{*,delta}`` after relocation.

The public ``evaluate_region_staleness.py`` harness calls ``run(config)``.
The callback returns one row per policy update and does not use the current
expert action to choose a deployed source.
"""
from __future__ import annotations

import copy
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch


def _arg(config: dict, name: str, default):
    return config.get(name, default)


def _history_obs(items):
    if not items:
        raise RuntimeError("empty observation history")
    result = {}
    for key in items[0]:
        values = [item[key] for item in items]
        values = [values[0]] * (len(items) - len(values)) + values
        result[key] = torch.stack(values, dim=1)
    return result


def _action_array(sequence, robot_name, joint_names):
    values = []
    for item in sequence:
        robot_action = item[robot_name]
        targets = robot_action.get("dof_pos_target", robot_action)
        values.append([targets[name] for name in joint_names])
    return np.asarray(values, dtype=np.float32)


def _chunk(actions: np.ndarray, start: int, horizon: int) -> np.ndarray:
    if len(actions) == 0:
        raise ValueError("expert action sequence is empty")
    ids = np.clip(np.arange(start, start + horizon), 0, len(actions) - 1)
    return actions[ids]


def _obs_dict(obs, robot_name):
    return {
        "rgb": obs.cameras["camera0"].rgb,
        "joint_qpos": obs.robots[robot_name].joint_pos,
    }


@torch.no_grad()
def run(config: dict):
    # The callback is executed on the server, where the original repo and
    # IsaacSim installation are available.
    repo = str(_arg(config, "repo", "/home/yiming/MomentVLA-main"))
    code_dir = str(Path(__file__).resolve().parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from flow_latent_predictor_common import (
        forward_flow,
        load_cache,
        load_flow,
        prepare_predictor_context,
        reverse_flow,
        seed_all,
    )

    seed = int(_arg(config, "seed", 0))
    seed_all(seed)
    device = str(_arg(config, "device", "cuda:0"))
    if not torch.cuda.is_available() and device.startswith("cuda"):
        device = "cpu"
    args = type("Args", (), {})()
    args.repo = repo
    args.checkpoint = str(_arg(config, "checkpoint", "/data/yiming/MomentVLA-main/il_outputs/fm_unet/pick_cube_isaac100/checkpoints/30.ckpt"))
    args.device = device
    args.cache = str(_arg(config, "cache", "/home/yiming/experiments/momentumvla/latent_predictor_diag_80_20_20260907/cache"))
    data, _ = load_cache(args.cache)
    policy, matcher, flow_cfg = load_flow(args)
    if int(_arg(config, "nfe", 200)) != 200:
        raise ValueError("formal staleness smoke requires 200 Flow steps")
    steps = 200
    scale = data["expert_raw"][data["split"] == 0].flatten(0, 1).std(0, unbiased=False).clamp_min(1e-6).to(device)

    def behavioral_exec_error(pred, target):
        pred_raw = policy.normalizer["action"].unnormalize(pred)
        target_raw = policy.normalizer["action"].unnormalize(target)
        normalized = (pred_raw - target_raw) / scale.to(pred.device)
        start = policy.n_obs_steps - 1
        end = start + stride
        return normalized[:, start:end].square().sum(dim=-1).mean(dim=-1).sqrt()

    from metasim.scenario.cameras import PinholeCameraCfg
    from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
    # Import the native task module explicitly.  Generic discovery can skip
    # it when one optional task package fails to import on a server worker.
    from roboverse_pack.tasks.maniskill import pick_cube as _pick_cube_task  # noqa: F401
    from metasim.task.registry import get_task_class
    from metasim.utils.demo_util import get_traj
    from metasim.utils.setup_util import get_robot

    task_name = str(_arg(config, "task", "pick_cube"))
    robot_name = str(_arg(config, "robot", "franka"))
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
    initial_states, all_expert_actions, _ = get_traj(env.traj_filepath, get_robot(robot_name), env.handler)
    joint_names = sorted(scenario.robots[0].joint_limits)
    episodes = min(int(_arg(config, "episodes", 1)), len(initial_states))
    shift_cm = float(_arg(config, "shift_cm", 0.0))
    horizon = int(policy.horizon)
    stride = int(policy.n_action_steps)
    rows = []

    def decode_error(z, condition, target_raw):
        target = policy.normalizer["action"].normalize(target_raw)
        decoded = forward_flow(policy, matcher, z, condition, steps, recompute=False)
        error = behavioral_exec_error(decoded, target)
        return float(error[0].detach().cpu())

    try:
        for ep in range(episodes):
            state = copy.deepcopy(initial_states[ep])
            state["objects"]["cube"]["pos"][0] += shift_cm / 100.0
            obs, _ = env.reset(states=[state])
            history = deque(maxlen=12)
            first = _obs_dict(obs, robot_name)
            history.append(first)
            actions = _action_array(all_expert_actions[ep], robot_name, joint_names)
            previous_z = None
            previous_condition = None
            query = 0
            # Replaying in stride-sized chunks makes the previous z correspond
            # to the immediately preceding executed action block.
            while query + stride < len(actions):
                obs_history = list(history)
                if len(obs_history) < 12:
                    obs_history = [obs_history[0]] * (12 - len(obs_history)) + obs_history
                encoded = prepare_predictor_context(policy, _history_obs(obs_history))
                condition = encoded[:, -policy.n_obs_steps:].flatten(1)
                current_raw = torch.from_numpy(_chunk(actions, query, horizon)).to(device).unsqueeze(0)
                current_norm = policy.normalizer["action"].normalize(current_raw)
                current_z = reverse_flow(policy, matcher, current_norm, condition, steps)
                if previous_z is not None:
                    e_reuse = decode_error(previous_z, condition, current_raw)
                    e_current = decode_error(current_z, condition, current_raw)
                    rows.append({
                        "episode": ep,
                        "query_index": query,
                        "e_reuse": e_reuse,
                        "e_current": e_current,
                        "reuse_minus_current": e_reuse - e_current,
                        "target_source": "recorded_demo_action_replay",
                        "shift_cm": shift_cm,
                    })
                previous_z = current_z.detach()
                previous_condition = condition.detach()
                for k in range(stride):
                    action_item = all_expert_actions[ep][min(query + k, len(all_expert_actions[ep]) - 1)]
                    obs, _, _, timeout, _ = env.step([action_item])
                    history.append(_obs_dict(obs, robot_name))
                    if bool(timeout[0]):
                        break
                query += stride
    finally:
        env.close()
    return rows
