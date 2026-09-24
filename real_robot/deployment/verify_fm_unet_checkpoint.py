#!/usr/bin/env python3
"""Verify that deployment preprocessing reproduces the training input exactly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from fm_unet_runtime import FMUnetRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    runtime = FMUnetRuntime(args.checkpoint, args.device)
    dataset_cfg = runtime.cfg.dataset_config.copy()
    dataset_cfg.zarr_path = args.zarr
    dataset = instantiate(dataset_cfg)
    raw = dataset[args.index]
    training_batch = dataset.postprocess(
        {key: value.unsqueeze(0) for key, value in raw.items()}, runtime.device
    )
    images = {
        obs_key: raw[data_key][: runtime.n_obs_steps].cpu().numpy()
        for data_key, obs_key in zip(dataset.image_data_keys, dataset.image_obs_keys)
    }
    deployed_obs = runtime.prepare_observation(
        images, raw[dataset.state_key][: runtime.n_obs_steps].cpu().numpy()
    )

    for key, expected in training_batch["obs"].items():
        # FM-UNet conditions only on the first n_obs_steps of each horizon.
        expected = expected[:, : runtime.n_obs_steps]
        actual = deployed_obs[key]
        if not torch.allclose(expected, actual, rtol=0.0, atol=1e-7):
            error = (expected - actual).abs().max().item()
            raise AssertionError(f"preprocessing mismatch for {key}: max_abs_error={error}")

    torch.manual_seed(123)
    expected_action = runtime.policy.predict_action(training_batch["obs"])["action"]
    torch.manual_seed(123)
    actual_action = runtime.policy.predict_action(deployed_obs)["action"]
    if not torch.allclose(expected_action, actual_action, rtol=1e-5, atol=1e-6):
        error = (expected_action - actual_action).abs().max().item()
        raise AssertionError(f"inference mismatch: max_abs_error={error}")

    print(
        f"verified checkpoint={args.checkpoint} state={runtime.state_name} "
        f"sample={args.index} action_shape={tuple(actual_action.shape)}"
    )


if __name__ == "__main__":
    main()
