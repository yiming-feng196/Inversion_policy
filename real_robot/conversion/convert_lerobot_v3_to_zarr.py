#!/usr/bin/env python3
"""Convert a local LeRobot v3 dataset into the flat Zarr used by FM-UNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-key", required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--video-backend", default=None)
    return parser.parse_args()


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def as_int(value) -> int:
    array = as_numpy(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def image_to_hwc_uint8(value, image_size: int) -> np.ndarray:
    image = as_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"RGB image must have three dimensions, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected three RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or Inf")
        if float(image.max(initial=0.0)) <= 1.0 + 1e-6:
            image = image * 255.0
    tensor = torch.as_tensor(np.ascontiguousarray(image)).permute(2, 0, 1).float()[None]
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return tensor[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def selected_indices(dataset: LeRobotDataset, stride: int) -> tuple[list[int], list[int]]:
    if stride < 1:
        raise ValueError("--frame-stride must be at least one")
    episode_ids = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    if episode_ids.ndim != 1 or len(episode_ids) != len(dataset):
        raise ValueError("Invalid episode_index column")
    indices: list[int] = []
    selected_episodes: list[int] = []
    for episode_id in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode_id)
        if rows.size == 0:
            continue
        keep = rows[::stride].tolist()
        if keep[-1] != int(rows[-1]):
            keep.append(int(rows[-1]))
        indices.extend(keep)
        selected_episodes.extend([int(episode_id)] * len(keep))
    if not indices:
        raise ValueError("The dataset contains no frames")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Dataset frames are not ordered by episode and time")
    return indices, selected_episodes


def create_array(group, name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype):
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def main() -> None:
    args = parse_args()
    info_path = args.input_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}. Choose a new path to avoid overwriting data."
        )
    if args.image_size <= 0 or args.chunk_frames <= 0:
        raise ValueError("Image size and chunk size must be positive")

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.input_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )
    required = {args.camera_key, args.state_key, args.action_key, "episode_index"}
    missing = sorted(required.difference(dataset.features))
    if missing:
        raise KeyError(f"LeRobot dataset is missing features: {missing}")

    indices, episode_ids = selected_indices(dataset, args.frame_stride)
    first = dataset[indices[0]]
    state_dim = int(as_numpy(first[args.state_key]).size)
    action_dim = int(as_numpy(first[args.action_key]).size)
    frame_count = len(indices)
    frame_chunk = min(args.chunk_frames, frame_count)

    root = zarr.open_group(str(args.output), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    image_array = create_array(
        data,
        "head_camera",
        (frame_count, args.image_size, args.image_size, 3),
        (frame_chunk, args.image_size, args.image_size, 3),
        "u1",
    )
    state_array = create_array(data, "state", (frame_count, state_dim), (frame_chunk, state_dim), "f4")
    action_array = create_array(data, "action", (frame_count, action_dim), (frame_chunk, action_dim), "f4")

    for output_index, source_index in enumerate(indices):
        item = dataset[source_index]
        state = as_numpy(item[args.state_key]).astype(np.float32, copy=False).reshape(-1)
        action = as_numpy(item[args.action_key]).astype(np.float32, copy=False).reshape(-1)
        if state.size != state_dim or action.size != action_dim:
            raise ValueError(
                f"Feature dimension changed at source frame {source_index}: "
                f"state {state.size}/{state_dim}, action {action.size}/{action_dim}"
            )
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"NaN or Inf at source frame {source_index}")
        image_array[output_index] = image_to_hwc_uint8(item[args.camera_key], args.image_size)
        state_array[output_index] = state
        action_array[output_index] = action

    episode_ends = np.flatnonzero(np.r_[np.diff(np.asarray(episode_ids)) != 0, True]) + 1
    meta.create_dataset("episode_ends", data=episode_ends.astype(np.int64), chunks=(len(episode_ends),))
    source_fps = float(dataset.meta.fps)
    manifest = {
        "format": "roboverse_flat_zarr_v1",
        "source_format": "lerobot_v3",
        "source_root": str(args.input_root.resolve()),
        "repo_id": args.repo_id,
        "camera_key": args.camera_key,
        "state_key": args.state_key,
        "action_key": args.action_key,
        "source_frames": int(len(dataset)),
        "output_frames": frame_count,
        "episodes": int(len(episode_ends)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": [args.image_size, args.image_size, 3],
        "frame_stride": args.frame_stride,
        "source_fps": source_fps,
        "effective_fps": source_fps / args.frame_stride,
    }
    root.attrs.update(manifest)
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
