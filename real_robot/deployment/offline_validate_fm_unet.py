#!/usr/bin/env python3
"""Compare an exact-checkpoint FM-UNet action chunk with recorded Zarr actions."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from hydra.utils import instantiate

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from fm_unet_runtime import FMUnetRuntime

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    runtime = FMUnetRuntime(args.checkpoint, args.device)
    dataset_cfg = runtime.cfg.dataset_config.copy()
    dataset_cfg.zarr_path = args.zarr
    dataset = instantiate(dataset_cfg)
    indices = np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int)
    predicted, recorded = [], []
    for ordinal, index in enumerate(indices):
        raw = dataset[int(index)]
        images = {
            obs_key: raw[data_key][: runtime.n_obs_steps].cpu().numpy()
            for data_key, obs_key in zip(dataset.image_data_keys, dataset.image_obs_keys)
        }
        predicted.append(runtime.predict(
            images, raw[dataset.state_key][: runtime.n_obs_steps].cpu().numpy(),
            seed=args.seed + ordinal,
        ))
        start = runtime.n_obs_steps - 1
        recorded.append(raw["action"][start : start + runtime.n_action_steps].cpu().numpy())
    predicted, recorded = np.stack(predicted), np.stack(recorded)
    residual = predicted - recorded
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "predictions.npz", sample_indices=indices,
                        predicted_action=predicted, recorded_action=recorded)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()), "state_dict": runtime.state_name,
        "zarr": args.zarr, "samples": int(len(indices)),
        "action_rmse": float(np.sqrt(np.mean(residual**2))),
        "action_mae": float(np.mean(np.abs(residual))),
        "hardware_commands_sent": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
