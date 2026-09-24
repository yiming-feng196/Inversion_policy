import copy
import datetime
import json
import os
import pathlib
import random
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import hydra
import imageio.v2 as iio
import numpy as np
import torch
import tqdm
import wandb
from loguru import logger as log

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.task.registry import get_task_class
from metasim.randomization import DomainRandomizationManager, DRConfig
from roboverse_learn.il.utils.ema_model import EMAModel
from roboverse_learn.il.runners.base_runner import BaseRunner
from roboverse_learn.il.utils.json_logger import JsonLogger
from roboverse_learn.il.utils.lr_scheduler import get_scheduler
from roboverse_learn.il.utils.pytorch_util import optimizer_to
from roboverse_learn.il.utils.visualization import plot_all_latent_visualizations

RANDOMIZATION_AVAILABLE = True


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def _snapshot_tensor_attributes(obj):
    """Clone mutable tensor fields used by task checkers during a branch.

    IsaacSim state restoration covers physics, but task checkers keep a small
    amount of Python-side state (for example ``DetectedChecker._first_check``).
    Counterfactual branches must not leak that state into the real rollout.
    """
    if obj is None:
        return {}
    return {
        name: value.clone()
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor)
    }


def _restore_tensor_attributes(obj, snapshot):
    if obj is None:
        return
    for name, value in snapshot.items():
        setattr(obj, name, value.clone())


def _stackcube_counterfactual_score(states, success):
    """Task-aware short-horizon oracle potential for StackCube.

    This score deliberately uses privileged simulator geometry.  It is an
    upper-bound diagnostic, not an inference-time method: before lift it
    rewards hand--cube approach; after lift it rewards cube motion toward the
    exact stack target.  Actual task success dominates every shaping term.
    """
    cube = states.objects["cube"].root_state[:, :3]
    base = states.objects["base"].root_state[:, :3]
    robot = states.robots["franka"]
    hand_index = robot.body_names.index("panda_hand")
    hand = robot.body_state[:, hand_index, :3]
    target = base.clone()
    target[:, 2] += 0.04
    ee_cube = torch.linalg.vector_norm(hand - cube, dim=-1)
    target_delta = cube - target
    target_error = torch.linalg.vector_norm(target_delta, dim=-1)
    target_linf_error = target_delta.abs().amax(dim=-1)
    xy_error = torch.linalg.vector_norm(cube[:, :2] - base[:, :2], dim=-1)
    relative_height = cube[:, 2] - base[:, 2]
    lifted = relative_height > 0.025

    # The stage offset makes a genuine lift preferable to a marginally closer
    # approach.  Once lifted, target distance is the task-relevant quantity.
    approach_score = -ee_cube + 2.0 * torch.clamp(relative_height, min=0.0)
    # RelativeBboxDetector is an axis-aligned +/-2 cm test.  L-infinity
    # distance is therefore the exact geometric margin to task success; the
    # small L2 term only breaks ties between equal checker margins.
    transport_score = (
        10.0 - 20.0 * target_linf_error - 1.0 * target_error
    )
    score = torch.where(lifted, transport_score, approach_score)
    success_tensor = torch.as_tensor(success, device=score.device, dtype=torch.bool)
    score = torch.where(success_tensor, torch.full_like(score, 100.0), score)
    metrics = {
        "ee_cube_distance": ee_cube.detach().cpu().tolist(),
        "cube_target_distance": target_error.detach().cpu().tolist(),
        "cube_target_linf_error": target_linf_error.detach().cpu().tolist(),
        "cube_base_xy_distance": xy_error.detach().cpu().tolist(),
        "cube_relative_height": relative_height.detach().cpu().tolist(),
        "lifted": lifted.detach().cpu().tolist(),
        "success": success_tensor.detach().cpu().tolist(),
    }
    return score, metrics


class DefaultRunner(BaseRunner):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.train_config.training_params.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy_config)
        self.policy_name = cfg.policy_name

        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.train_config.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.eval_args = hydra.utils.instantiate(cfg.eval_config.eval_args)

    def train(self):
        cfg = copy.deepcopy(self.cfg)
        # When resuming, optionally interpret num_epochs as the number of
        # additional epochs instead of restarting the original schedule.
        resume_additional_epochs = cfg.train_config.training_params.get(
            "resume_additional_epochs", None
        )

        # resume training
        if cfg.train_config.training_params.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if (
            cfg.train_config.training_params.resume
            and resume_additional_epochs is not None
        ):
            run_num_epochs = int(resume_additional_epochs)
            # Checkpoints are written before self.epoch is incremented. Move
            # to the next epoch so the resumed 100.ckpt starts at epoch 100.
            self.epoch += 1
            scheduler_num_epochs = self.epoch + run_num_epochs
        else:
            run_num_epochs = int(cfg.train_config.training_params.num_epochs)
            scheduler_num_epochs = run_num_epochs

        # Optional global optimizer-step budget. ``max_train_steps`` is a
        # per-epoch cap in this runner, so use this separate setting when a
        # run is specified in total steps (e.g. 50k steps with batch size 128).
        max_total_train_steps = cfg.train_config.training_params.get(
            "max_total_train_steps", None
        )
        if max_total_train_steps is not None:
            max_total_train_steps = int(max_total_train_steps)

        # configure dataset
        dataset = hydra.utils.instantiate(cfg.dataset_config)
        train_dataloader = create_dataloader(dataset, **cfg.train_config.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(
            val_dataset, **cfg.train_config.val_dataloader
        )

        self.model.set_normalizer(normalizer)
        if cfg.train_config.training_params.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        scheduler_steps = len(train_dataloader) * scheduler_num_epochs
        if max_total_train_steps is not None:
            scheduler_steps = max_total_train_steps
        lr_scheduler = get_scheduler(
            cfg.train_config.training_params.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.train_config.training_params.lr_warmup_steps,
            num_training_steps=(
                scheduler_steps
                // cfg.train_config.training_params.gradient_accumulate_every
            ),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.train_config.training_params.use_ema:
            ema = hydra.utils.instantiate(cfg.train_config.ema, model=self.ema_model)

        wandb_run = None

        # configure logging
        if cfg.logging.mode == "online":
            # Truncate tags to max 64 characters (wandb limit)
            logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            if "tags" in logging_cfg and logging_cfg["tags"]:
                logging_cfg["tags"] = [tag[:64] if len(tag) > 64 else tag for tag in logging_cfg["tags"]]

            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **logging_cfg,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # device transfer
        device = torch.device(cfg.train_config.training_params.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.train_config.training_params.debug:
            cfg.train_config.training_params.num_epochs = 2
            cfg.train_config.training_params.max_train_steps = 3
            cfg.train_config.training_params.max_val_steps = 3
            cfg.train_config.training_params.rollout_every = 1
            cfg.train_config.training_params.checkpoint_every = 1
            cfg.train_config.training_params.val_every = 1
            cfg.train_config.training_params.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        save_best_checkpoint = bool(
            cfg.train_config.training_params.get("save_best_checkpoint", False)
        )
        best_val_loss = float("inf")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(run_num_epochs):
                if (
                    max_total_train_steps is not None
                    and self.global_step >= max_total_train_steps
                ):
                    break
                step_log = dict()
                stop_after_epoch = False
                # ========= train for this epoch ==========
                if cfg.train_config.training_params.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        if (
                            max_total_train_steps is not None
                            and self.global_step >= max_total_train_steps
                        ):
                            stop_after_epoch = True
                            break
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = (
                            raw_loss
                            / cfg.train_config.training_params.gradient_accumulate_every
                        )
                        loss.backward()

                        # Update the optimizer only after a complete gradient
                        # accumulation window. EMA must follow the *updated*
                        # parameters, never intermediate gradients.
                        is_optimizer_step = (
                            (batch_idx + 1)
                            % cfg.train_config.training_params.gradient_accumulate_every
                            == 0
                        )
                        if is_optimizer_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            if cfg.train_config.training_params.use_ema:
                                ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        # Optional policy metrics (e.g. nominal_innovation). Backward-compatible:
                        # A2A has no last_metrics attribute, so this is a no-op for the baseline.
                        if hasattr(self.model, "last_metrics") and self.model.last_metrics:
                            step_log.update(self.model.last_metrics)

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.train_config.training_params.max_train_steps is not None
                        ) and batch_idx >= (
                            cfg.train_config.training_params.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.train_config.training_params.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.train_config.training_params.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.train_config.training_params.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                # Select checkpoints with the same policy that
                                # is evaluated and deployed.  For EMA runs this
                                # must be the EMA model, not the raw model.
                                loss = policy.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.train_config.training_params.max_val_steps
                                    is not None
                                ) and batch_idx >= (
                                    cfg.train_config.training_params.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss
                            if save_best_checkpoint and val_loss < best_val_loss:
                                best_val_loss = val_loss
                                checkpoint_dir = pathlib.Path(
                                    cfg.checkpoint.save_root_dir
                                ).joinpath("checkpoints")
                                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                                temporary_path = checkpoint_dir.joinpath("best.ckpt.tmp")
                                best_path = checkpoint_dir.joinpath("best.ckpt")
                                # Block until the exact state evaluated above is safely
                                # written, then atomically publish it as the new best.
                                self.save_checkpoint(temporary_path, use_thread=False)
                                os.replace(temporary_path, best_path)
                                metadata_path = checkpoint_dir.joinpath("best.json")
                                metadata_tmp = checkpoint_dir.joinpath("best.json.tmp")
                                metadata_tmp.write_text(
                                    json.dumps(
                                        {
                                            "epoch": self.epoch + 1,
                                            "val_loss": best_val_loss,
                                            "checkpoint": str(best_path),
                                        },
                                        indent=2,
                                    )
                                    + "\n"
                                )
                                os.replace(metadata_tmp, metadata_path)
                                log.info(
                                    f"Saved best checkpoint at epoch {self.epoch + 1} "
                                    f"with val_loss={best_val_loss:.6f}"
                                )

                # Latent space visualization for A2A policy
                if hasattr(policy, 'get_latents_for_visualization'):
                    try:
                        with torch.no_grad():
                            # Collect latents from multiple validation batches
                            all_history_latents = []
                            all_future_latents = []
                            max_samples = 500  # Limit samples for t-SNE performance
                            first_batch = None

                            for batch_idx, batch in enumerate(val_dataloader):
                                batch = dataset.postprocess(batch, device)
                                if first_batch is None:
                                    first_batch = batch  # Save for trajectory visualization
                                history_latents, future_latents = policy.get_latents_for_visualization(batch)
                                all_history_latents.append(history_latents.cpu())
                                all_future_latents.append(future_latents.cpu())

                                if sum(h.shape[0] for h in all_history_latents) >= max_samples:
                                    break

                            # Concatenate all collected latents
                            history_latents = torch.cat(all_history_latents, dim=0)[:max_samples]
                            future_latents = torch.cat(all_future_latents, dim=0)[:max_samples]

                            # Get flow trajectories for visualization (uses model's num_sampling_steps)
                            trajectories = None
                            trajectory_targets = None
                            if hasattr(policy, 'get_flow_trajectories') and first_batch is not None:
                                trajectories, trajectory_targets = policy.get_flow_trajectories(
                                    first_batch, n_samples=5
                                )

                            # Generate all visualizations
                            viz_dir = pathlib.Path(self.output_dir) / "latent_viz"
                            viz_results = plot_all_latent_visualizations(
                                history_latents=history_latents,
                                future_latents=future_latents,
                                epoch=self.epoch + 1,
                                save_dir=str(viz_dir),
                                trajectories=trajectories,
                                trajectory_targets=trajectory_targets,
                            )
                            log.info(f"Saved latent visualizations to {viz_dir}")
                            log.info(f"  Avg t-SNE Distance: {viz_results['avg_tsne_distance']:.2f}")

                            # Log metrics to wandb
                            wandb_metrics = {
                                "latent/avg_tsne_distance": viz_results['avg_tsne_distance'],
                            }
                            if 'flow_end_to_target_dist' in viz_results:
                                wandb_metrics["latent/flow_end_to_target_dist"] = viz_results['flow_end_to_target_dist']
                            if wandb_run is not None:
                                wandb_run.log(wandb_metrics, step=self.global_step)
                    except Exception as e:
                        log.warning(f"Failed to generate latent visualization: {e}")

                # run diffusion sampling on a training batch
                if (
                    (self.epoch % cfg.train_config.training_params.sample_every) == 0
                    and not getattr(policy, "is_recovery_policy", False)
                ):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]

                        # Handle shape mismatch (e.g., VITA action-to-action flow outputs 8 frames from horizon=16)
                        pred_len = pred_action.shape[1]
                        gt_len = gt_action.shape[1]
                        if pred_len != gt_len:
                            # For action-to-action flow: pred is future actions starting from n_obs_steps-1
                            # Slice gt_action to match: take the corresponding future portion
                            n_obs_steps = gt_len - pred_len + 1  # Infer n_obs_steps from shape difference
                            start_idx = n_obs_steps - 1
                            gt_action = gt_action[:, start_idx:start_idx + pred_len, :]

                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (
                    (self.epoch + 1) % cfg.train_config.training_params.checkpoint_every
                ) == 0 or self.epoch + 1 >= scheduler_num_epochs:
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.dataset_config.zarr_path).stem
                    self.save_checkpoint(
                        cfg.checkpoint.save_root_dir
                        + f"/checkpoints/{self.epoch + 1}.ckpt"
                    )  # TODO

                if (
                    max_total_train_steps is not None
                    and self.global_step >= max_total_train_steps
                ):
                    stop_after_epoch = True

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                # The epoch-end increment accounts for the final batch when
                # the per-epoch loop naturally reaches its end.  If a global
                # step budget stopped the loop after a non-final batch, that
                # batch already incremented global_step; avoid counting one
                # extra step here.
                if not (
                    max_total_train_steps is not None
                    and self.global_step >= max_total_train_steps
                ):
                    self.global_step += 1
                self.epoch += 1
                if stop_after_epoch:
                    break

    def evaluate(self, ckpt_path=None):
        args = self.eval_args

        # Checkpoints store the policy weights/configuration, but the dataset
        # normalizer is fitted at training time and is not part of the model
        # state dict. Evaluation is commonly launched in a fresh process, so
        # restore the exact training normalizer before predict_action().
        # Without this, observations/actions are interpreted in raw
        # coordinates (identity normalization), which can make an otherwise
        # valid policy fail every episode.
        eval_dataset = hydra.utils.instantiate(self.cfg.dataset_config)
        eval_normalizer = eval_dataset.get_normalizer()
        self.model.set_normalizer(eval_normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(eval_normalizer)
        del eval_dataset

        # Enable timeout-guarded close to avoid IsaacSim shutdown hang
        import os
        os.environ["METASIM_FORCE_EXIT_ON_CLOSE"] = "1"
        os.environ.setdefault("METASIM_CLOSE_TIMEOUT_SEC", "8")

        num_envs: int = args.num_envs
        log.info(f"Using GPU device: {args.gpu_id}")
        task_cls = get_task_class(args.task)

        # Camera configuration
        if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
            dp_camera = True
        else:
            dp_camera = args.task != "close_box"

        is_libero_dataset = "libero_90" in args.task

        if is_libero_dataset:
            dp_pos = (2.0, 0.0, 2)
        elif dp_camera:
            dp_pos = (1.0, 0.0, 0.75)
        else:
            dp_pos = (1.5, 0.0, 1.5)

        # Match the image resolution and camera count expected by the
        # checkpoint.  RLBench image policies use front/agentview plus an
        # eye-in-hand camera; the latter is mounted to the Panda hand in the
        # simulator so it follows the robot during rollout.
        two_view_eval = "eye_in_hand_cam" in self.cfg.shape_meta.obs
        if two_view_eval:
            image_shape = self.cfg.shape_meta.obs.agentview_cam.shape
            image_height, image_width = int(image_shape[-2]), int(image_shape[-1])
        else:
            image_height, image_width = 256, 256
        camera = PinholeCameraCfg(
            name="camera0",
            data_types=["rgb", "depth"],
            width=image_width,
            height=image_height,
            pos=dp_pos,
            look_at=(0.0, 0.0, 0.0),
        )
        cameras = [camera]
        if two_view_eval:
            cameras.append(
                PinholeCameraCfg(
                    name="camera1",
                    data_types=["rgb"],
                    width=image_width,
                    height=image_height,
                    mount_to=args.robot,
                    mount_link="panda_hand",
                    mount_pos=(0.05, 0.0, 0.08),
                    mount_quat=(1.0, 0.0, 0.0, 0.0),
                )
            )

        # Lighting setup
        render_mode = getattr(args, 'render_mode', 'raytracing')
        if render_mode == "pathtracing":
            ceiling_main = 18000.0
            ceiling_corners = 8000.0
        else:
            ceiling_main = 12000.0
            ceiling_corners = 5000.0

        from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
        lights = [
            DiskLightCfg(
                name="ceiling_main",
                intensity=ceiling_main,
                color=(1.0, 1.0, 1.0),
                radius=1.2,
                pos=(0.0, 0.0, 2.8),
                rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
            SphereLightCfg(
                name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
            ),
            SphereLightCfg(
                name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
            ),
            SphereLightCfg(
                name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
            ),
            SphereLightCfg(
                name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
            ),
        ]

        scenario = task_cls.scenario.update(
            robots=[args.robot],
            simulator=args.sim,
            num_envs=args.num_envs,
            headless=args.headless,
            lights=lights,
            cameras=cameras
        )

        flip_images = bool(getattr(args, "flip_images", False))

        def eval_rgb(current_obs, camera_name):
            image = current_obs.cameras[camera_name].rgb
            if flip_images:
                image = torch.flip(image, dims=[1])
            return image
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tic = time.time()
        env = task_cls(scenario, device=device)
        robot = get_robot(args.robot)

        # Domain Randomization configuration
        dr_level = getattr(args, 'level', 0)
        dr_scene_mode = getattr(args, 'scene_mode', 0)
        dr_seed = getattr(args, 'randomization_seed', None)

        mujoco_level1_randomizer = None
        if args.sim == "mujoco" and dr_level == 1:
            # The generic Scene/Material randomizers operate on IsaacSim USD
            # and MDL assets.  Use the repository's rendering-only MuJoCo
            # equivalent so level-1 keeps physics and initial states unchanged.
            from roboverse_learn.il.policies.a2a.eval_flow_reversal_shift import (
                MujocoLevel1Randomizer,
            )

            mujoco_level1_randomizer = MujocoLevel1Randomizer(
                env.handler,
                seed=42 if dr_seed is None else dr_seed,
            )
            randomization_manager = None
            log.info(
                "Domain Randomization enabled: level=1, "
                "backend=mujoco_material_texture, "
                f"seed={42 if dr_seed is None else dr_seed}"
            )
        elif not RANDOMIZATION_AVAILABLE:
            if dr_level > 0:
                log.warning("Domain randomization requested but not available!")
            randomization_manager = None
        else:
            from dataclasses import dataclass as dc

            @dc
            class SimpleRenderCfg:
                mode: str = render_mode

            randomization_manager = DomainRandomizationManager(
                config=DRConfig(
                    level=dr_level,
                    scene_mode=dr_scene_mode,
                    randomization_seed=dr_seed,
                ),
                scenario=scenario,
                handler=env.handler,
                init_states=None,
                render_cfg=SimpleRenderCfg(mode=render_mode)
            )
            if dr_level > 0:
                log.info(f"Domain Randomization enabled: level={dr_level}, scene_mode={dr_scene_mode}, seed={dr_seed}")
            else:
                log.info("Domain Randomization disabled (level=0)")

        toc = time.time()
        log.trace(f"Time to launch: {toc - tic:.2f}s")

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = self.get_checkpoint_path()
        checkpoint = ckpt_path if ckpt_path is not None else checkpoint
        if checkpoint is None:
            raise ValueError(
                "No checkpoint found, please provide a valid checkpoint path."
            )
        args.checkpoint_path = pathlib.Path(checkpoint)
        ckpt_name = args.checkpoint_path.name + "_" + time_str
        ckpt_name = f"{args.task}/{self.policy_name}/{args.robot}/{ckpt_name}"

        eval_runner_class = self.get_eval_runner_class()
        policyRunner = eval_runner_class(
            self,
            scenario=scenario,
            num_envs=num_envs,
            checkpoint_path=args.checkpoint_path,
            device=f"cuda:{args.gpu_id}",
            task_name=args.task,
            subset=args.subset,
        )

        action_set_steps = (
            2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
        )
        # Data
        tic = time.time()
        assert os.path.exists(env.traj_filepath), (
            f"Trajectory file: {env.traj_filepath} does not exist."
        )
        init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
        num_demos = len(init_states)
        toc = time.time()
        log.trace(f"Time to load data: {toc - tic:.2f}s")

        # Optional initial-state cube shift for robustness evaluation.  With
        # the default (0, 0) this is a no-op and preserves the baseline
        # DefaultEvalRunner protocol exactly.
        cube_shift_x = float(getattr(args, "cube_shift_x", 0.0))
        cube_shift_y = float(getattr(args, "cube_shift_y", 0.0))
        cube_shift_object = getattr(args, "cube_shift_object", "cube")
        cube_shift_step = int(getattr(args, "cube_shift_step", -1))
        if (cube_shift_x != 0.0 or cube_shift_y != 0.0) and cube_shift_step < 0:
            shifted = 0
            for init_state in init_states:
                objects = init_state.get("objects", {})
                obj_state = objects.get(cube_shift_object)
                if obj_state is None:
                    log.warning(
                        f"cube shift requested for object={cube_shift_object!r}, "
                        "but it is absent from an initial state"
                    )
                    continue
                obj_state["pos"][0] += cube_shift_x
                obj_state["pos"][1] += cube_shift_y
                shifted += 1
            log.info(
                f"Cube shift enabled: object={cube_shift_object}, "
                f"dx={cube_shift_x:.4f}m, dy={cube_shift_y:.4f}m, "
                f"shifted_states={shifted}/{num_demos}"
            )
        elif (cube_shift_x != 0.0 or cube_shift_y != 0.0) and cube_shift_step >= 0:
            log.info(
                f"Scheduled dynamic cube shift: object={cube_shift_object}, "
                f"after_env_step={cube_shift_step}, "
                f"dx={cube_shift_x:.4f}m, dy={cube_shift_y:.4f}m"
            )

        # Update DR manager with init_states
        if randomization_manager is not None:
            randomization_manager.init_states = init_states
            randomization_manager.original_positions = {}
            for demo_idx, init_state in enumerate(init_states):
                demo_key = f"demo_{demo_idx}"
                randomization_manager.original_positions[demo_key] = {}

                if "objects" in init_state:
                    for obj_name, obj_state in init_state["objects"].items():
                        randomization_manager.original_positions[demo_key][f"obj_{obj_name}"] = {
                            "x": float(obj_state["pos"][0]),
                            "y": float(obj_state["pos"][1]),
                            "z": float(obj_state["pos"][2]),
                        }

                if "robots" in init_state:
                    for robot_name, robot_state in init_state["robots"].items():
                        randomization_manager.original_positions[demo_key][f"robot_{robot_name}"] = {
                            "x": float(robot_state["pos"][0]),
                            "y": float(robot_state["pos"][1]),
                            "z": float(robot_state["pos"][2]),
                        }

        total_success = 0
        total_completed = 0
        all_inference_times = []  # Collect inference times from all steps
        demo_avg_inference_times = []  # Collect average inference time for each demo

        if args.max_demo is None:
            max_demos = args.task_id_range_high - args.task_id_range_low
        else:
            max_demos = args.max_demo
        max_demos = min(max_demos, num_demos)

        for demo_start_idx in range(
            args.task_id_range_low, args.task_id_range_low + max_demos, num_envs
        ):
            demo_end_idx = min(demo_start_idx + num_envs, num_demos)
            current_demo_idxs = list(range(demo_start_idx, demo_end_idx))

            # Bank2 episode-increment protocol: re-seed every demo block with
            # base_seed + demo_start_idx so policy/stochastic sources behave
            # like one independent run per episode (the controller protocol).
            if num_envs == 1:
                base_seed = int(getattr(args, "randomization_seed", 4242) or 4242)
                episode_seed = base_seed + demo_start_idx
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(episode_seed)
                log.info(f"[DP Eval] Episode {demo_start_idx}: RNG seed = {episode_seed}")

            # Apply domain randomization before reset
            if mujoco_level1_randomizer is not None:
                mujoco_level1_randomizer.apply(demo_start_idx)
                log.info(f"[DP Eval] Episode {demo_start_idx}: Applying MuJoCo Level-1 DR")
            elif randomization_manager is not None and dr_level > 0:
                for env_id, demo_idx in enumerate(current_demo_idxs):
                    log.info(f"[DP Eval] Episode {demo_idx}: Applying DR")
                    randomization_manager.apply_randomization(
                        demo_idx=demo_idx, is_initial=(demo_start_idx == args.task_id_range_low))
                    randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=env_id)
                    randomization_manager.update_camera_look_at(env_id=env_id)
                    randomization_manager.apply_camera_randomization()

            tic = time.time()
            obs, extras = env.reset(states=init_states[demo_start_idx:demo_end_idx])
            toc = time.time()
            log.trace(f"Time to reset: {toc - tic:.2f}s")

            # Ensure environment stabilizes after reset
            if randomization_manager is not None and dr_level > 0:
                ensure_clean_state(env.handler)

                if hasattr(env, "_episode_steps"):
                    for env_id in range(num_envs):
                        env._episode_steps[env_id] = 0

            policyRunner.reset()

            step = 0
            MaxStep = args.max_step
            counterfactual_oracle = bool(
                getattr(args, "counterfactual_source_oracle", False)
            )
            counterfactual_branch_steps = max(
                1, int(getattr(args, "counterfactual_branch_steps", 8))
            )
            episode_trajectory_oracle = bool(
                getattr(args, "episode_trajectory_oracle", False)
            )
            episode_trajectory_candidates = max(
                1, int(getattr(args, "episode_trajectory_candidates", 4))
            )
            if counterfactual_oracle and episode_trajectory_oracle:
                raise ValueError(
                    "Local counterfactual oracle and episode trajectory oracle "
                    "cannot be enabled together"
                )
            if counterfactual_oracle:
                if args.task != "stack_cube":
                    raise ValueError(
                        "counterfactual_source_oracle currently has a task-aware "
                        "potential only for stack_cube"
                    )
                if num_envs != 1:
                    raise ValueError(
                        "counterfactual_source_oracle requires num_envs=1 for "
                        "exact paired state branching"
                    )
                if not hasattr(policyRunner, "prepare_counterfactual_candidates"):
                    raise TypeError(
                        "Selected evaluation runner does not expose discrete "
                        "counterfactual source candidates"
                    )
            SuccessOnce = [False] * num_envs
            TimeOut = [False] * num_envs
            cube_shift_applied = False
            images_list = []
            inference_times = []  # Record inference time for each step
            print(policyRunner.policy_cfg)

            if episode_trajectory_oracle:
                if num_envs != 1:
                    raise ValueError(
                        "episode_trajectory_oracle requires num_envs=1 for exact "
                        "paired full-rollout branches"
                    )
                if not hasattr(policyRunner, "flow_qkv_episode_candidate_rank"):
                    raise TypeError(
                        "Selected policy runner does not expose an episode-level "
                        "inverse trajectory candidate rank"
                    )
                if (
                    cube_shift_x != 0.0
                    or cube_shift_y != 0.0
                    or cube_shift_step >= 0
                ):
                    raise ValueError(
                        "The first episode trajectory oracle test supports only "
                        "the clean protocol"
                    )

                # Every candidate starts from the exact same simulator and
                # checker state.  Keeping all branches in one IsaacSim process
                # removes cross-process rendering/physics drift from Oracle@K.
                initial_obs = copy.deepcopy(obs)
                physics_snapshot = copy.deepcopy(
                    env.handler.get_states(mode="tensor")
                )
                episode_steps_snapshot = env._episode_steps.clone()
                checker = getattr(env, "checker", None)
                checker_snapshot = _snapshot_tensor_attributes(checker)
                detector = getattr(checker, "detector", None)
                detector_snapshot = _snapshot_tensor_attributes(detector)
                policy_episode_counter = int(policyRunner._episode_counter)

                def restore_episode_snapshot():
                    env.handler.set_states(
                        states=copy.deepcopy(physics_snapshot), env_ids=[0]
                    )
                    env._episode_steps = episode_steps_snapshot.clone()
                    _restore_tensor_attributes(checker, checker_snapshot)
                    _restore_tensor_attributes(detector, detector_snapshot)
                    env.handler.refresh_render()

                branch_rows = []
                branch_images = []
                branch_inference_times = []
                for candidate_rank in range(episode_trajectory_candidates):
                    restore_episode_snapshot()
                    # VisualPhaseSkillBankRunner.reset() advances its episode
                    # counter. Rewind by one so every branch retains the same
                    # LODO query ID and fixed candidate schedule.
                    policyRunner._episode_counter = policy_episode_counter - 1
                    policyRunner.flow_qkv_episode_candidate_rank = candidate_rank
                    policyRunner.reset()

                    candidate_obs = copy.deepcopy(initial_obs)
                    candidate_step = 0
                    candidate_success_once = False
                    candidate_success_end = False
                    candidate_timeout = False
                    candidate_images = []
                    candidate_times = []
                    first_expert_episode = None
                    first_global_start = None

                    while candidate_step < MaxStep:
                        candidate_new_obs = {
                            "rgb": eval_rgb(candidate_obs, "camera0"),
                            "joint_qpos": candidate_obs.robots[
                                args.robot
                            ].joint_pos,
                        }
                        if two_view_eval:
                            candidate_new_obs["agentview_rgb"] = candidate_new_obs["rgb"]
                            candidate_new_obs["eye_in_hand_rgb"] = eval_rgb(
                                candidate_obs, "camera1"
                            )
                        candidate_images.append(
                            np.array(candidate_new_obs["rgb"].cpu())
                        )
                        inference_start = time.time()
                        candidate_action = policyRunner.get_action(
                            candidate_new_obs
                        )
                        candidate_times.append(
                            (time.time() - inference_start) * 1000.0
                        )
                        if first_expert_episode is None:
                            first_expert_episode = int(
                                policyRunner._locked_episode[0]
                            )
                            first_global_start = int(
                                policyRunner._locked_start[0]
                            )

                        for _ in range(action_set_steps):
                            (
                                candidate_obs,
                                reward,
                                candidate_success,
                                candidate_time_out,
                                extras,
                            ) = env.step(candidate_action)

                        candidate_success_end = bool(
                            candidate_success[0].item()
                        )
                        candidate_timeout = bool(
                            candidate_time_out[0].item()
                        )
                        candidate_success_once = (
                            candidate_success_once or candidate_success_end
                        )
                        candidate_step += 1
                        if candidate_success_once:
                            break

                    branch_rows.append(
                        {
                            "task_index": int(demo_start_idx),
                            "candidate_rank": int(candidate_rank),
                            "expert_episode": first_expert_episode,
                            "expert_global_start": first_global_start,
                            "success_once": bool(candidate_success_once),
                            "success_end": bool(candidate_success_end),
                            "time_out": bool(candidate_timeout),
                            "steps": int(candidate_step),
                            "mean_inference_ms": float(
                                np.mean(candidate_times)
                            ) if candidate_times else 0.0,
                        }
                    )
                    branch_images.append(candidate_images)
                    branch_inference_times.append(candidate_times)
                    log.info(
                        "Episode trajectory branch: "
                        f"task={demo_start_idx}, rank={candidate_rank}, "
                        f"expert_episode={first_expert_episode}, "
                        f"start={first_global_start}, "
                        f"success={candidate_success_once}, "
                        f"steps={candidate_step}"
                    )

                successful_ranks = [
                    row["candidate_rank"]
                    for row in branch_rows
                    if row["success_once"]
                ]
                selected_rank = successful_ranks[0] if successful_ranks else 0
                selected_row = branch_rows[selected_rank]
                images_list = branch_images[selected_rank]
                inference_times = [
                    value
                    for candidate_times in branch_inference_times
                    for value in candidate_times
                ]
                SuccessOnce = [bool(successful_ranks)]
                TimeOut = [bool(selected_row["time_out"])]
                success = torch.tensor(
                    [selected_row["success_end"]],
                    dtype=torch.bool,
                    device=env.device,
                )
                time_out = torch.tensor(
                    [selected_row["time_out"]],
                    dtype=torch.bool,
                    device=env.device,
                )
                step = int(selected_row["steps"])
                # Skip the ordinary single-policy rollout below: all K
                # complete candidate trajectories have already been run.
                MaxStep = 0
                oracle_log_path = pathlib.Path(self.output_dir).joinpath(
                    "episode_trajectory_oracle.jsonl"
                )
                with oracle_log_path.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "task_index": int(demo_start_idx),
                                "candidate_count": episode_trajectory_candidates,
                                "selected_rank": int(selected_rank),
                                "successful_ranks": successful_ranks,
                                "oracle_success": bool(successful_ranks),
                                "branches": branch_rows,
                            }
                        )
                        + "\n"
                    )

            while step < MaxStep:
                new_obs = {
                    "rgb": eval_rgb(obs, "camera0"),
                    "joint_qpos": obs.robots[args.robot].joint_pos,
                }
                if two_view_eval:
                    new_obs["agentview_rgb"] = new_obs["rgb"]
                    new_obs["eye_in_hand_rgb"] = eval_rgb(obs, "camera1")

                images_list.append(np.array(new_obs["rgb"].cpu()))

                # Measure inference time
                inference_start = time.time()
                if counterfactual_oracle and len(policyRunner.action_cache) == 0:
                    candidate_package = policyRunner.prepare_counterfactual_candidates(
                        new_obs
                    )
                    candidate_chunks = candidate_package["qpos_chunks"]
                    physics_snapshot = copy.deepcopy(
                        env.handler.get_states(mode="tensor")
                    )
                    episode_steps_snapshot = env._episode_steps.clone()
                    checker = getattr(env, "checker", None)
                    checker_snapshot = _snapshot_tensor_attributes(checker)
                    detector = getattr(checker, "detector", None)
                    detector_snapshot = _snapshot_tensor_attributes(detector)
                    branch_scores = []
                    branch_metrics = []

                    def restore_counterfactual_snapshot():
                        env.handler.set_states(
                            states=copy.deepcopy(physics_snapshot), env_ids=[0]
                        )
                        env._episode_steps = episode_steps_snapshot.clone()
                        _restore_tensor_attributes(checker, checker_snapshot)
                        _restore_tensor_attributes(detector, detector_snapshot)
                        env.handler.refresh_render()

                    for candidate_chunk in candidate_chunks:
                        restore_counterfactual_snapshot()
                        branch_success = torch.zeros(
                            num_envs, dtype=torch.bool, device=env.device
                        )
                        for candidate_qpos in candidate_chunk[
                            :counterfactual_branch_steps
                        ]:
                            branch_action = policyRunner.action_to_dict(
                                candidate_qpos
                            )
                            _, _, candidate_success, _, _ = env.step(branch_action)
                            branch_success |= candidate_success
                        branch_state = env.handler.get_states(mode="tensor")
                        score, metrics = _stackcube_counterfactual_score(
                            branch_state, branch_success
                        )
                        branch_scores.append(float(score[0].item()))
                        branch_metrics.append(metrics)

                    restore_counterfactual_snapshot()
                    selected_candidate = int(np.argmax(branch_scores))
                    action = policyRunner.commit_counterfactual_candidate(
                        selected_candidate,
                        branch_scores=branch_scores,
                        branch_metrics=branch_metrics,
                    )
                else:
                    action = policyRunner.get_action(new_obs)
                inference_end = time.time()
                inference_time_ms = (inference_end - inference_start) * 1000
                inference_times.append(inference_time_ms)

                log.debug(f"Step {step} | Inference time: {inference_time_ms:.2f}ms")

                for round_i in range(action_set_steps):
                    obs, reward, success, time_out, extras = env.step(action)

                # Optional online perturbation.  This is deliberately placed
                # after an environment step, so the next observation window
                # contains the moved cube and the policy receives genuine
                # closed-loop feedback.  With cube_shift_step=-1 this block is
                # a no-op and the historical evaluation protocol is unchanged.
                if (
                    cube_shift_step >= 0
                    and not cube_shift_applied
                    and step + 1 == cube_shift_step
                    and (cube_shift_x != 0.0 or cube_shift_y != 0.0)
                ):
                    shifted_state = env.handler.get_states(mode="tensor")
                    if cube_shift_object not in shifted_state.objects:
                        raise KeyError(
                            f"Dynamic cube shift object {cube_shift_object!r} "
                            "is absent from simulator state"
                        )
                    root_state = shifted_state.objects[cube_shift_object].root_state
                    root_state[:, 0] += cube_shift_x
                    root_state[:, 1] += cube_shift_y
                    env.handler.set_states(
                        states=shifted_state,
                        env_ids=list(range(num_envs)),
                    )
                    env.handler.refresh_render()
                    cube_shift_applied = True
                    log.info(
                        f"Applied dynamic cube shift at env_step={step + 1}: "
                        f"object={cube_shift_object}, dx={cube_shift_x:.4f}m, "
                        f"dy={cube_shift_y:.4f}m"
                    )

                # eval
                SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
                TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
                step += 1
                if all(SuccessOnce):
                    break

            # Calculate inference time statistics
            total_steps = len(inference_times)
            avg_inference_time = sum(inference_times) / total_steps if total_steps > 0 else 0
            min_inference_time = min(inference_times) if inference_times else 0
            max_inference_time = max(inference_times) if inference_times else 0

            log.info(f"Demo {demo_start_idx}-{demo_end_idx}: Avg inference time: {avg_inference_time:.2f}ms, "
                     f"Min: {min_inference_time:.2f}ms, Max: {max_inference_time:.2f}ms, Total steps: {total_steps}")

            # Collect inference times for overall statistics
            all_inference_times.extend(inference_times)
            demo_avg_inference_times.append(avg_inference_time)  # Store demo-level average

            SuccessEnd = success.tolist()
            total_success += SuccessOnce.count(True)
            total_completed += len(SuccessOnce)
            base_eval_dir = pathlib.Path(self.output_dir).joinpath("eval", ckpt_name)
            base_eval_dir.mkdir(parents=True, exist_ok=True)
            for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
                demo_idx_str = str(demo_idx).zfill(4)
                if args.save_video_freq > 0 and demo_idx % args.save_video_freq == 0:
                    iio.mimwrite(
                        str(base_eval_dir.joinpath(f"{demo_idx}.mp4")),
                        [images[i] for images in images_list],
                    )
                with open(base_eval_dir.joinpath(f"{demo_idx_str}.txt"), "w") as f:
                    f.write(f"Demo Index: {demo_idx}\n")
                    f.write(f"Num Envs: {num_envs}\n")
                    f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                    f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                    f.write(f"TimeOut: {TimeOut[i]}\n")
                    f.write(f"Domain Randomization Level: {dr_level}\n")
                    f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
                    f.write(f"Domain Randomization Seed: {dr_seed}\n")
                    f.write(
                        f"Cumulative Average Success Rate: {total_success / total_completed:.4f}\n"
                    )
                    # Add inference time statistics
                    f.write(f"\n--- Inference Time Statistics ---\n")
                    f.write(f"Total Steps: {total_steps}\n")
                    f.write(f"Average Inference Time: {avg_inference_time:.2f}ms\n")
                    f.write(f"Min Inference Time: {min_inference_time:.2f}ms\n")
                    f.write(f"Max Inference Time: {max_inference_time:.2f}ms\n")
            log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
            log.info("Num Envs: ", num_envs)
            log.info(f"SuccessOnce: {SuccessOnce}")
            log.info(f"SuccessEnd: {SuccessEnd}")
            log.info(f"TimeOut: {TimeOut}")
        # Calculate overall inference time statistics
        overall_total_steps = len(all_inference_times)
        overall_avg_inference_time = sum(all_inference_times) / overall_total_steps if overall_total_steps > 0 else 0
        overall_min_inference_time = min(all_inference_times) if all_inference_times else 0
        overall_max_inference_time = max(all_inference_times) if all_inference_times else 0

        # Calculate STD of demo-level average inference times
        num_demos_evaluated = len(demo_avg_inference_times)
        if num_demos_evaluated > 1:
            demo_avg_mean = sum(demo_avg_inference_times) / num_demos_evaluated
            demo_avg_variance = sum((x - demo_avg_mean) ** 2 for x in demo_avg_inference_times) / (num_demos_evaluated - 1)
            demo_avg_std = demo_avg_variance ** 0.5
        else:
            demo_avg_std = 0.0

        log.info(f"FINAL RESULTS: Average Success Rate = {total_success / total_completed:.4f}")
        log.info(f"FINAL RESULTS: Overall Avg Inference Time = {overall_avg_inference_time:.2f}ms (STD across demos: {demo_avg_std:.2f}ms), "
                 f"Min: {overall_min_inference_time:.2f}ms, Max: {overall_max_inference_time:.2f}ms, "
                 f"Total Steps: {overall_total_steps}")

        with open(base_eval_dir.joinpath("final_stats.txt"), "w") as f:
            f.write(f"=== Success Statistics ===\n")
            f.write(f"Total Success: {total_success}\n")
            f.write(f"Total Completed: {total_completed}\n")
            f.write(f"Average Success Rate: {total_success / total_completed:.4f}\n")
            f.write(f"\n=== Domain Randomization ===\n")
            f.write(f"Domain Randomization Level: {dr_level}\n")
            f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
            f.write(f"Domain Randomization Seed: {dr_seed}\n")
            f.write(f"\n=== Overall Inference Time Statistics ===\n")
            f.write(f"Total Inference Steps: {overall_total_steps}\n")
            f.write(f"Number of Demos Evaluated: {num_demos_evaluated}\n")
            f.write(f"Average Inference Time: {overall_avg_inference_time:.2f}ms\n")
            f.write(f"STD of Demo Avg Inference Time: {demo_avg_std:.2f}ms\n")
            f.write(f"Min Inference Time: {overall_min_inference_time:.2f}ms\n")
            f.write(f"Max Inference Time: {overall_max_inference_time:.2f}ms\n")
        if mujoco_level1_randomizer is not None:
            mujoco_level1_randomizer.restore()
        env.close()

    @staticmethod
    def get_eval_runner_class():
        """Return the rollout adapter used by :meth:`evaluate`.

        Specialized policies can override this without duplicating the common
        MetaSim evaluation loop and success accounting.
        """
        from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner

        return DefaultEvalRunner

    def run(
        self,
        train=None,
        eval=None,
        ckpt_path=None,
    ):
        train = self.cfg.train_enable
        eval = self.cfg.eval_enable
        # Always use eval_path if provided (respects num_epochs setting)
        ckpt_path = self.cfg.eval_path
        if train:
            self.train()
        if eval:
            self.evaluate(ckpt_path=ckpt_path)


class BatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
        weights=None,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None
        self.weights = None
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (data_size,) or np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("sampling weights must be non-negative with shape [data_size]")
            self.weights = weights / weights.sum()

    def __iter__(self):
        if self.shuffle:
            if self.weights is None:
                perm = self.rng.permutation(self.data_size)
            else:
                # Weighted sampling with replacement keeps the epoch length
                # fixed while exposing rare transition windows more often.
                perm = self.rng.choice(
                    self.data_size,
                    size=self.num_batch * self.batch_size,
                    replace=True,
                    p=self.weights,
                )
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0 and self.weights is None:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    # print("create_dataloader_batch_size", batch_size)
    batch_sampler = BatchSampler(
        len(dataset),
        batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
        weights=(getattr(dataset, "sampling_weights", None) if shuffle else None),
    )

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = DefaultRunner(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
