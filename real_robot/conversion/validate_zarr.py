#!/usr/bin/env python3
"""Validate a real-robot flat Zarr before FM-UNet training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def finite_min_max(array, chunk_size: int) -> tuple[list[float], list[float]]:
    minimum = np.full(array.shape[1], np.inf, dtype=np.float64)
    maximum = np.full(array.shape[1], -np.inf, dtype=np.float64)
    for start in range(0, array.shape[0], chunk_size):
        block = np.asarray(array[start : start + chunk_size])
        if not np.isfinite(block).all():
            raise ValueError(f"{array.path} contains NaN or Inf near frame {start}")
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
    return minimum.tolist(), maximum.tolist()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    root = zarr.open_group(str(args.zarr), mode="r")
    required = ("data/head_camera", "data/state", "data/action", "meta/episode_ends")
    missing = [key for key in required if key not in root]
    if missing:
        raise KeyError(f"Missing Zarr arrays: {missing}")

    image = root["data/head_camera"]
    state = root["data/state"]
    action = root["data/action"]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    frame_count = int(action.shape[0])
    if image.shape[0] != frame_count or state.shape[0] != frame_count:
        raise ValueError("Image, state, and action frame counts differ")
    if frame_count == 0 or ends.size == 0 or int(ends[-1]) != frame_count:
        raise ValueError("episode_ends must be non-empty and end at the total frame count")
    if np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("Every episode must contain at least one frame")
    if image.dtype != np.dtype("uint8") or image.ndim != 4:
        raise ValueError(f"head_camera must be a uint8 rank-4 RGB array, got {image.dtype} {image.shape}")
    if image.shape[-1] == 3:
        image_layout = "NHWC"
    elif image.shape[1] == 3:
        image_layout = "NCHW"
    else:
        raise ValueError(f"head_camera has no three-channel RGB axis: {image.shape}")
    if state.ndim != 2 or action.ndim != 2:
        raise ValueError("state and action must be two-dimensional")

    state_min, state_max = finite_min_max(state, args.chunk_size)
    action_min, action_max = finite_min_max(action, args.chunk_size)
    action_max_step = np.zeros(action.shape[1], dtype=np.float64)
    begin = 0
    for end in ends:
        episode = np.asarray(action[begin:int(end)])
        if len(episode) > 1:
            action_max_step = np.maximum(action_max_step, np.abs(np.diff(episode, axis=0)).max(axis=0))
        begin = int(end)

    report = {
        "status": "valid",
        "zarr": str(args.zarr.resolve()),
        "frames": frame_count,
        "episodes": int(len(ends)),
        "episode_length_min": int(np.diff(np.r_[0, ends]).min()),
        "episode_length_max": int(np.diff(np.r_[0, ends]).max()),
        "image_shape": list(image.shape[1:]),
        "image_layout": image_layout,
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "state_min": state_min,
        "state_max": state_max,
        "action_min": action_min,
        "action_max": action_max,
        "action_max_abs_step": action_max_step.tolist(),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
