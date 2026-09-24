#!/usr/bin/env python3
"""Exact-checkpoint runtime for real-robot Flow-Matching UNet policies.

The checkpoint config defines the visual encoder, image normalization, action
horizon, and EMA selection. This module intentionally does not import robot
drivers or send actions to hardware.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FMUnetRuntime:
    """Build an inference-only FM-UNet directly from a training checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda:0"):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.payload = torch.load(
            self.checkpoint, map_location="cpu", pickle_module=__import__("dill")
        )
        self.cfg = self.payload["cfg"]
        states = self.payload.get("state_dicts", {})
        self.state_name = "ema_model" if "ema_model" in states else "model"
        if self.state_name not in states:
            raise ValueError("checkpoint has neither an EMA nor an online model state")

        # The checkpoint completely overwrites visual weights. Avoid a network
        # request for ImageNet initialization when loading on the robot.
        policy_cfg = copy.deepcopy(self.cfg.policy_config)
        rgb_cfg = policy_cfg.obs_encoder.get("rgb_model", None)
        if rgb_cfg is not None and "weights" in rgb_cfg:
            rgb_cfg.weights = None
        self.policy = instantiate(policy_cfg)
        self.policy.load_state_dict(states[self.state_name], strict=True)
        self.policy.eval().to(self.device)

        self.n_obs_steps = int(self.cfg.n_obs_steps)
        self.n_action_steps = int(self.cfg.n_action_steps)
        self.image_obs_keys = [
            key
            for key, spec in self.cfg.shape_meta.obs.items()
            if spec.type == "rgb"
        ]
        self.lowdim_obs_keys = [
            key
            for key, spec in self.cfg.shape_meta.obs.items()
            if spec.type != "rgb"
        ]
        if len(self.lowdim_obs_keys) != 1:
            raise ValueError(
                "runtime currently expects exactly one low-dimensional observation key, "
                f"got {self.lowdim_obs_keys}"
            )
        self.state_obs_key = self.lowdim_obs_keys[0]
        self.image_shapes = {
            key: tuple(int(v) for v in self.cfg.shape_meta.obs[key].shape)
            for key in self.image_obs_keys
        }
        self.state_dim = int(self.cfg.shape_meta.obs[self.state_obs_key].shape[0])
        self.action_dim = int(self.cfg.shape_meta.action.shape[0])

    def _prepare_rgb(self, frames: np.ndarray, obs_key: str) -> torch.Tensor:
        """Convert chronological RGB uint8 HWC frames to [1,T,C,H,W] floats.

        Frame order and RGB channel order are preserved. Image mean/std
        normalization is performed only by the checkpoint's normalizer.
        """
        frames = np.asarray(frames)
        expected_c, expected_h, expected_w = self.image_shapes[obs_key]
        if frames.ndim != 4 or frames.shape[0] != self.n_obs_steps:
            raise ValueError(
                f"{obs_key}: expected {self.n_obs_steps} chronological frames, got {frames.shape}"
            )
        if frames.shape[-1] != expected_c:
            raise ValueError(
                f"{obs_key}: expected HWC RGB with {expected_c} channels, got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            raise TypeError(
                f"{obs_key}: expected uint8 RGB frames before normalization, got {frames.dtype}"
            )
        image = torch.from_numpy(np.ascontiguousarray(frames))
        image = image.permute(0, 3, 1, 2).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        if tuple(image.shape[-2:]) != (expected_h, expected_w):
            image = F.interpolate(
                image.flatten(0, 1).float(),
                size=(expected_h, expected_w),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (1, self.n_obs_steps))
        # Match RobotImageDataset.postprocess exactly: cast and divide after
        # moving uint8 HWC data to the inference device.
        return image.float().div_(255.0)

    def prepare_observation(
        self,
        images: Mapping[str, np.ndarray],
        states: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        missing = set(self.image_obs_keys) - set(images)
        unexpected = set(images) - set(self.image_obs_keys)
        if missing or unexpected:
            raise KeyError(f"image keys mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        state = np.asarray(states, dtype=np.float32)
        if state.shape != (self.n_obs_steps, self.state_dim):
            raise ValueError(
                f"{self.state_obs_key}: expected {(self.n_obs_steps, self.state_dim)}, got {state.shape}"
            )
        obs = {key: self._prepare_rgb(images[key], key) for key in self.image_obs_keys}
        obs[self.state_obs_key] = torch.from_numpy(
            np.ascontiguousarray(state)
        ).unsqueeze(0).to(self.device, non_blocking=True)
        return obs

    @torch.inference_mode()
    def predict(
        self,
        images: Mapping[str, np.ndarray],
        states: np.ndarray,
        seed: int | None = None,
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        obs = self.prepare_observation(images, states)
        action = self.policy.predict_action(obs)["action"]
        action = action.squeeze(0).detach().cpu().numpy()
        if action.shape != (self.n_action_steps, self.action_dim):
            raise RuntimeError(
                f"unexpected action shape {action.shape}; expected "
                f"{(self.n_action_steps, self.action_dim)}"
            )
        if not np.isfinite(action).all():
            raise FloatingPointError("FM-UNet produced non-finite actions")
        return action


def _read_npz_input(path: Path, runtime: FMUnetRuntime) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path) as data:
        images = {key: np.asarray(data[key]) for key in runtime.image_obs_keys}
        states = np.asarray(data[runtime.state_obs_key])
    return images, states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="NPZ containing RGB keys from shape_meta and the low-dimensional state key.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runtime = FMUnetRuntime(args.checkpoint, args.device)
    images, states = _read_npz_input(args.input, runtime)
    action = runtime.predict(images, states, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, action=action)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "state_dict": runtime.state_name,
        "image_obs_keys": runtime.image_obs_keys,
        "state_obs_key": runtime.state_obs_key,
        "action_shape": list(action.shape),
        "hardware_commands_sent": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
