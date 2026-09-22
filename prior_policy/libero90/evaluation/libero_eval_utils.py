"""CPU-only checks for paired evaluation of the existing LIBERO action flow.

No environment, checkpoint, or GPU dependency. These helpers deliberately do not
declare packaged initial-state indices to be held-out demonstration IDs.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np


def _finite(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}, got {value.shape}")
    return value


def quat_to_axisangle(quat):
    """Match the xyzw convention used by the existing LIBERO runner.

    Do not canonicalize quaternion signs: that would differ from the established
    runner. Validate this conversion against stored ee_states before rollout.
    """
    q = _finite(quat, (4,), "xyzw quaternion").astype(np.float64).copy()
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
        raise ValueError("Expected a unit xyzw quaternion")
    q[3] = np.clip(q[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - q[3] ** 2))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * (2 * math.acos(q[3])) / denominator).astype(np.float32)


def raw_frame(obs, environment_step):
    """Return rotated RGB and raw 8-D state, matching Corpus preprocessing."""
    cameras = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = _finite(obs[key], (128, 128, 3), key)
        if image.dtype != np.uint8:
            raise ValueError("Render at 128px uint8; do not silently resize or rescale")
        cameras.append(np.ascontiguousarray(image[::-1, ::-1]))
    state = np.concatenate([
        _finite(obs["robot0_eef_pos"], (3,), "eef position"),
        quat_to_axisangle(obs["robot0_eef_quat"]),
        _finite(obs["robot0_gripper_qpos"], (2,), "gripper qpos"),
    ]).astype(np.float32)
    return {"step": int(environment_step), "images": np.stack(cameras), "state": state}


def _norm_parameters(normalizer, dimension):
    center = _finite(normalizer["center"], (dimension,), "normalizer center").astype(np.float32)
    scale = _finite(normalizer["scale"], (dimension,), "normalizer scale").astype(np.float32)
    if np.any(scale <= 0):
        raise ValueError("Normalizer scales must be positive")
    return center, scale


def policy_inputs(history, state_normalizer):
    """Two consecutive env observations -> [1,2,2,3,128,128], [1,2,8].

    Call raw_frame after EVERY env step, not only every policy query. ImageNet
    normalization is performed later by Policy.condition, not by this helper.
    """
    if len(history) != 2 or history[1]["step"] - history[0]["step"] != 1:
        raise ValueError("History must contain exactly two consecutive environment frames")
    images = np.stack([_finite(f["images"], (2, 128, 128, 3), "images") for f in history])
    if images.dtype != np.uint8:
        raise ValueError("History images must remain uint8 until conversion")
    state = np.stack([_finite(f["state"], (8,), "state") for f in history])
    center, scale = _norm_parameters(state_normalizer, 8)
    images = images.transpose(0, 1, 4, 2, 3)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(images), ((state - center) / scale)[None].astype(np.float32)


def executable_actions(normalized_chunk, action_normalizer, action_bounds=None):
    """Decode training normalization, then select [1:9]; no gripper sign flip.

    Optional environment action-spec clipping is applied uniformly to all
    samplers and its frequency must be logged. There is NO clipping in normalized
    coordinates, which would change the trained decoder and exclude valid tails.
    """
    chunk = _finite(normalized_chunk, (16, 7), "normalized action chunk").astype(np.float32)
    center, scale = _norm_parameters(action_normalizer, 7)
    actions = (chunk * scale + center)[1:9].copy()
    clipped_coordinates = 0
    if action_bounds is not None:
        lower = _finite(action_bounds[0], (7,), "action lower bounds")
        upper = _finite(action_bounds[1], (7,), "action upper bounds")
        if np.any(lower > upper):
            raise ValueError("Invalid action bounds")
        clipped_coordinates = int(np.count_nonzero((actions < lower) | (actions > upper)))
        actions = np.clip(actions, lower, upper).astype(np.float32)
    return actions, {"clipped_coordinates": clipped_coordinates, "coordinates": int(actions.size)}


def paired_noise_seed(task, initial_state_index, policy_seed, query_index):
    """A method-independent counter seed avoids RNG desynchronization by work."""
    if min(initial_state_index, policy_seed, query_index) < 0:
        raise ValueError("Indices and seeds must be nonnegative")
    key = f"q2-v1|{task}|{initial_state_index}|{policy_seed}|{query_index}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1)


def paired_success_summary(records, methods, bootstrap_repeats=2000, seed=20260919):
    """Strict paired rates and initial-state-cluster bootstrap differences.

    Required record keys: method, task, initial_state_index, env_seed,
    policy_seed, status='complete', success (bool), protocol_id. protocol_id must
    hash shared environment/decoder settings and frozen checkpoint, not sampler
    parameters. No missing/error episodes are silently discarded or converted.
    """
    if len(methods) < 2 or len(set(methods)) != len(methods):
        raise ValueError("Supply at least two distinct methods in reporting order")
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    by_task = defaultdict(lambda: defaultdict(dict))
    protocol_by_task = {}
    for r in records:
        if r["method"] not in methods or r["status"] != "complete" or type(r["success"]) is not bool:
            raise ValueError("Unexpected method, incomplete trial, or non-boolean success")
        task, protocol = r["task"], r["protocol_id"]
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("Each record requires a nonempty shared protocol_id")
        if task in protocol_by_task and protocol_by_task[task] != protocol:
            raise ValueError("Mismatched decoder/environment protocol within task")
        protocol_by_task[task] = protocol
        key = (int(r["initial_state_index"]), int(r["env_seed"]), int(r["policy_seed"]))
        if key in by_task[task][r["method"]]:
            raise ValueError("Duplicate method/trial key")
        by_task[task][r["method"]][key] = r["success"]
    if not by_task:
        raise ValueError("No records")
    rng = np.random.default_rng(seed)
    tasks = {}
    for task, values in sorted(by_task.items()):
        keys = set(values[methods[0]])
        if not keys or any(set(values[m]) != keys for m in methods):
            raise ValueError(f"Methods do not have identical complete paired cases for {task}")
        ordered = sorted(keys)
        clusters = sorted({k[0] for k in ordered})
        report = {"trials_per_method": len(ordered), "independent_initial_states": len(clusters),
                  "protocol_id": protocol_by_task[task], "methods": {}, "paired_differences": []}
        for method in methods:
            outcomes = [values[method][k] for k in ordered]
            report["methods"][method] = {"successes": int(sum(outcomes)), "trials": len(outcomes),
                                        "success_rate": float(np.mean(outcomes))}
        for i, first in enumerate(methods):
            for second in methods[i+1:]:
                a = np.array([values[first][k] for k in ordered], dtype=float)
                b = np.array([values[second][k] for k in ordered], dtype=float)
                cluster_delta = np.array([np.mean([values[first][k] - int(values[second][k])
                                                   for k in ordered if k[0] == state]) for state in clusters])
                samples = cluster_delta[rng.integers(len(clusters), size=(bootstrap_repeats, len(clusters)))].mean(1)
                report["paired_differences"].append({"first": first, "second": second,
                    "trial_mean_difference": float(np.mean(a-b)),
                    "initial_state_mean_difference": float(cluster_delta.mean()),
                    "ci95_initial_state_bootstrap": np.quantile(samples, [.025, .975]).tolist(),
                    "first_only_successes": int(np.count_nonzero((a == 1) & (b == 0))),
                    "second_only_successes": int(np.count_nonzero((a == 0) & (b == 1)))})
        tasks[task] = report
    return {"tasks": tasks, "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": seed,
            "scope": "Paired rollout evidence on recorded initial states; not an unseen-state guarantee."}
