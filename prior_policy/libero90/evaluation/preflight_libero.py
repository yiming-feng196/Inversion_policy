"""Restore an actual HDF5 demonstration before trusting live FM preprocessing.

This is a data/environment convention check, not a learned-policy evaluation.
No dummy settling is performed after restoring the recorded simulator state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from eval_closed_loop import digest_file, resolve_task, write_json
from libero_eval_utils import raw_frame


def image_comparison(live, stored):
    if live.shape != stored.shape or live.dtype != np.uint8 or stored.dtype != np.uint8:
        raise ValueError(f"Image mismatch {live.shape}/{stored.shape}, {live.dtype}/{stored.dtype}")
    candidates = {"identity": stored, "vertical_flip": stored[::-1],
                  "horizontal_flip": stored[:, ::-1], "rotate_180": stored[::-1, ::-1]}
    errors = {name: float(np.sqrt(np.mean(((live.astype(np.float64) - candidate) / 255.)**2)))
              for name, candidate in candidates.items()}
    return {"rmse_0_1": errors, "best_stored_transform": min(errors, key=errors.get),
            "identity_excess_over_best": errors["identity"] - min(errors.values())}


def find_env_attribute(env, name):
    """LIBERO ControlEnv wraps a robosuite environment in .env."""
    current, seen = env, set()
    for _ in range(4):
        if id(current) in seen:
            break
        seen.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "env", None)
        if current is None:
            break
    raise AttributeError(f"No {name} on LIBERO wrapper/underlying environment")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--demo", default="demo_0")
    p.add_argument("--task-name", required=True)
    p.add_argument("--suite", default="libero_90", choices=["libero_goal", "libero_90"])
    p.add_argument("--replay-steps", type=int, default=32)
    p.add_argument("--alignment-frames", type=int, default=3)
    p.add_argument("--gate-mode", choices=["replay", "state_convention"], default="replay",
                   help="Replay requires exact recorded dynamics; state_convention checks joint/gripper state, live quaternion encoding, and camera orientation")
    p.add_argument("--dataset-creation-script", type=Path,
                   help="Optional exact LIBERO/scripts/create_dataset.py path, hashed as timing-convention evidence")
    p.add_argument("--use-demo-model", action="store_true",
                   help="Restore trusted dataset's model_file XML as well as state; never rewrite dataset assets")
    p.add_argument("--max-state-error", type=float, default=1e-4)
    p.add_argument("--max-image-rmse", type=float, default=.10)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.alignment_frames < 1 or args.replay_steps < 0:
        raise ValueError("Invalid number of alignment/replay frames")
    if args.output.exists():
        raise FileExistsError("Use a new preflight output directory")
    args.output.mkdir(parents=True)
    report = {"status": "running", "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "scope": "Dataset simulator-state replay; no policy success claim", "data_sha256": digest_file(args.hdf5)}
    destination = args.output / "preflight.json"
    write_json(destination, report)
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    env = None
    try:
        with h5py.File(args.hdf5, "r") as f:
            demo = f["data"][args.demo]
            states = np.asarray(demo["states"])
            actions = np.asarray(demo["actions"], np.float32)
            target_states = np.concatenate([np.asarray(demo["obs"]["ee_states"]),
                                             np.asarray(demo["obs"]["gripper_states"])], -1)
            target_joint_states = np.asarray(demo["obs"]["joint_states"])
            alignment_count = min(args.alignment_frames, len(states)-1, len(target_states))
            if alignment_count < 1:
                raise ValueError("Need a subsequent saved simulator state to check post-action observation timing")
            saved_images = [np.asarray(demo["obs"][key][:alignment_count]) for key in ("agentview_rgb", "eye_in_hand_rgb")]
            model_xml = demo.attrs.get("model_file")
            if isinstance(model_xml, bytes):
                model_xml = model_xml.decode()
        suite = benchmark.get_benchmark_dict()[args.suite]()
        task_id, task = resolve_task(suite, args.task_name)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        report.update(task_id=task_id, bddl_file=str(bddl), bddl_sha256=digest_file(bddl),
                      demo_actions_shape=list(actions.shape), demo_states_shape=list(states.shape),
                      stored_model_xml_available=model_xml is not None,
                      gate_mode=args.gate_mode,
                      observation_alignment={
                          "dataset_convention": "create_dataset.py saves env.step(action_j) observation as obs[j], alongside pre-action states[j] and actions[j]",
                          "gate_alignment": "restore states[0], replay raw actions[j], compare env.step observations to stored obs[j] for first requested frames",
                          "restored_post_action_control": "states[j+1] vs obs[j] remains a diagnostic; explicit sim.forward may refresh kinematics relative to cached env.step observations",
                          "pre_action_control": "restore states[0] vs obs[0] is recorded but not a matching-state gate",
                          "limitation": "This validates observation/state conversion under documented dataset timing; it does not correct or certify causal BC target alignment.",
                          "base_checkpoint_cache_and_online_action_indexing_changed": False})
        if args.gate_mode == "state_convention":
            report["observation_alignment"].update(
                gate_alignment="restore states[j+1], compare live robot joints/gripper to stored obs[j]; compare axis-angle encoder to official robosuite on the SAME live quaternion; compare cameras",
                limitation="Encoding-convention check only. Saved simulator state does not encode all controller/gripper history; exact dynamic replay, stale-vs-refreshed FK equality and policy success are not certified.")
        if args.dataset_creation_script is not None:
            report["observation_alignment"]["creation_script"] = str(args.dataset_creation_script)
            report["observation_alignment"]["creation_script_sha256"] = digest_file(args.dataset_creation_script)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
        env.seed(7)
        env.reset()
        if args.use_demo_model:
            if not isinstance(model_xml, str) or not model_xml.strip().startswith("<"):
                raise ValueError("Dataset has no valid model_file XML")
            find_env_attribute(env, "reset_from_xml_string")(model_xml)
        def compare_observation(obs, stored_index):
            frame = raw_frame(obs, stored_index)
            state_delta = frame["state"] - target_states[stored_index]
            images = {saved_key: image_comparison(obs[live_key], stored[stored_index])
                      for live_key, saved_key, stored in zip(
                          ("agentview_image", "robot0_eye_in_hand_image"),
                          ("agentview_rgb", "eye_in_hand_rgb"), saved_images)}
            return {"stored_observation_index": stored_index,
                    "state_max_abs_error": float(np.abs(state_delta).max()),
                    "state_rmse": float(np.sqrt(np.mean(state_delta**2))),
                    "live_state": frame["state"].tolist(), "dataset_state": target_states[stored_index].tolist(),
                    "image_comparisons": images}

        pre_obs = env.set_init_state(states[0])
        report["pre_action_vs_post_action_control"] = compare_observation(pre_obs, 0)
        aligned = []
        from robosuite.utils.transform_utils import quat2axisangle as official_quat2axisangle
        for index in range(alignment_count):
            obs = env.set_init_state(states[index+1])
            row = compare_observation(obs, index)
            row["restored_simulator_state_index"] = index+1
            live_joint = np.asarray(obs["robot0_joint_pos"])
            if live_joint.shape != target_joint_states[index].shape:
                raise ValueError("Live/stored joint position dimensions differ")
            official_axisangle = np.asarray(official_quat2axisangle(np.asarray(obs["robot0_eef_quat"]).copy()))
            own_axisangle = raw_frame(obs, index)["state"][3:6]
            row["encoding_convention"] = {
                "joint_max_abs_error": float(np.abs(live_joint-target_joint_states[index]).max()),
                "gripper_max_abs_error": float(np.abs(np.asarray(obs["robot0_gripper_qpos"])-target_states[index,6:]).max()),
                "axisangle_vs_official_same_quaternion_max_abs_error": float(np.abs(own_axisangle-official_axisangle).max()),
                "live_joint_positions": live_joint.tolist(), "stored_joint_positions": target_joint_states[index].tolist(),
                "axisangle_reference": "robosuite.utils.transform_utils.quat2axisangle, evaluated on identical live robot0_eef_quat"}
            aligned.append(row)
            if index == 0:
                np.savez_compressed(args.output / "initial_observation.npz", live_state=np.asarray(row["live_state"]),
                    dataset_state=target_states[0], live_agentview=obs["agentview_image"],
                    live_wrist=obs["robot0_eye_in_hand_image"], dataset_agentview=saved_images[0][0],
                    dataset_wrist=saved_images[1][0], stored_observation_index=0, restored_simulator_state_index=1)
        lower, upper = find_env_attribute(env, "action_spec")
        report.update(action_spec={"lower": np.asarray(lower).tolist(), "upper": np.asarray(upper).tolist()},
                      aligned_post_action_checks=aligned,
                      restored_post_action_state_max_abs_error=max(row["state_max_abs_error"] for row in aligned))

        def restore_start():
            # Fresh controller state matters for action replay; pure state
            # restoration alone need not reset controller goal/history caches.
            env.reset()
            if args.use_demo_model:
                find_env_attribute(env, "reset_from_xml_string")(model_xml)
            return env.set_init_state(states[0])

        # Dataset extraction itself stores observations returned by env.step.
        # Reproduce that process rather than loosening a tolerance on a different
        # observation-refresh path. The static state-restoration controls stay.
        restore_start()
        replay_aligned = []
        for index in range(alignment_count):
            obs, _, replay_done, _ = env.step(actions[index].tolist())
            row = compare_observation(obs, index)
            row.update(action_index=index, done=bool(replay_done))
            replay_aligned.append(row)
            if index == 0:
                np.savez_compressed(args.output / "step_observation.npz", live_state=np.asarray(row["live_state"]),
                    dataset_state=target_states[0], live_agentview=obs["agentview_image"],
                    live_wrist=obs["robot0_eye_in_hand_image"], dataset_agentview=saved_images[0][0],
                    dataset_wrist=saved_images[1][0], stored_observation_index=0, executed_action_index=0)
        report.update(replay_post_action_checks=replay_aligned,
                      state_max_abs_error=max(row["state_max_abs_error"] for row in replay_aligned),
                      state_rmse=float(np.mean([row["state_rmse"] for row in replay_aligned])),
                      live_state=replay_aligned[0]["live_state"], dataset_state=replay_aligned[0]["dataset_state"],
                      image_comparisons={f"obs{row['stored_observation_index']}/{camera}": values
                                         for row in replay_aligned for camera,values in row["image_comparisons"].items()})
        if args.gate_mode == "state_convention":
            report.update(
                state_max_abs_error=max(row["state_max_abs_error"] for row in aligned),
                state_rmse=float(np.mean([row["state_rmse"] for row in aligned])),
                live_state=aligned[0]["live_state"], dataset_state=aligned[0]["dataset_state"],
                image_comparisons={f"obs{row['stored_observation_index']}/{camera}": values
                                  for row in aligned for camera,values in row["image_comparisons"].items()},
                encoding_gate_metrics={key: max(row["encoding_convention"][key] for row in aligned)
                    for key in ("joint_max_abs_error", "gripper_max_abs_error", "axisangle_vs_official_same_quaternion_max_abs_error")},
                encoding_gate_tolerance=1e-6,
                eef_state_error_scope="Diagnostic only: exact joints/gripper but refreshed live FK can differ from stored cached end-effector observation",
                replay_scope="Diagnostic only: simulator snapshots do not restore every controller/gripper internal state")
        # Separate reset ensures this diagnostic cannot change the gated replay.
        restore_start()
        step_obs, _, _, _ = env.step(actions[0].tolist())
        step_frame = raw_frame(step_obs, 0)
        try:
            simulation_state = find_env_attribute(env, "get_sim_state")()
            refreshed_obs = find_env_attribute(env, "regenerate_obs_from_state")(simulation_state)
            refreshed_frame = raw_frame(refreshed_obs, 0)
            difference = refreshed_frame["state"] - step_frame["state"]
            report["kinematic_refresh_diagnostic"] = {
                "status": "complete", "procedure": "After first raw action, compare returned env.step obs with regenerate_obs_from_state(get_sim_state())",
                "step_vs_dataset": compare_observation(step_obs, 0),
                "refreshed_vs_dataset": compare_observation(refreshed_obs, 0),
                "refreshed_minus_step_state": difference.tolist(),
                "refreshed_minus_step_max_abs": float(np.abs(difference).max()),
                "state_blocks_max_abs": {"eef_position": float(np.abs(difference[:3]).max()),
                                         "axis_angle": float(np.abs(difference[3:6]).max()),
                                         "gripper": float(np.abs(difference[6:]).max())}}
        except (AttributeError, TypeError):
            report["kinematic_refresh_diagnostic"] = {"status": "unavailable", "error": traceback.format_exc(),
                "scope": "No explanation of restored-state residual is asserted when refresh API is unavailable"}
        image_gate = all(row["rmse_0_1"]["identity"] <= args.max_image_rmse and
                         row["identity_excess_over_best"] <= 1e-3 for row in report["image_comparisons"].values())
        if args.gate_mode == "state_convention":
            gate = image_gate and all(value <= 1e-6 for value in report["encoding_gate_metrics"].values())
        else:
            gate = image_gate and report["state_max_abs_error"] <= args.max_state_error
        report["preprocessing_gate_passed"] = bool(gate)
        write_json(destination, report)
        if not gate:
            raise RuntimeError(f"Preprocessing {args.gate_mode} check failed; do not run learned comparisons")
        trajectory = []
        done = False
        # Reset back to the pre-action state; no residual state from the checks.
        restore_start()
        for index in range(min(args.replay_steps, len(actions))):
            obs, _, done, _ = env.step(actions[index].tolist())
            row = {"action_index": index, "done": bool(done)}
            if index < len(target_states):
                live_state = raw_frame(obs, index + 1)["state"]
                row["stored_post_action_observation_index"] = index
                row["post_action_state_max_abs_error"] = float(np.abs(live_state-target_states[index]).max())
                row["post_action_state_rmse"] = float(np.sqrt(np.mean((live_state-target_states[index])**2)))
            trajectory.append(row)
            if done:
                break
        report.update(status="complete", replay=trajectory, replay_success=bool(done),
                      note="Finite action replay is a convention sanity check, not a complete policy result.")
        write_json(destination, report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        write_json(destination, report)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
