"""Derive a temporally thinned inversion cache from a full cache without reinversion."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=32)
    return parser.parse_args()


def selected_sample_ids(source: Path, manifest: dict, stride: int) -> set[int]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for name in manifest["shards"]:
        shard = torch.load(source / name, map_location="cpu", weights_only=True)
        for sample_id, split, episode, sampler_index in zip(
            shard["sample_index"].tolist(),
            shard["split"].tolist(),
            shard["episode"].tolist(),
            shard["sampler_index"].tolist(),
        ):
            groups[(int(split), int(episode))].append((int(sampler_index), int(sample_id)))

    selected: set[int] = set()
    for rows in groups.values():
        rows.sort()
        selected.update(sample_id for _, sample_id in rows[::stride])
        selected.add(rows[-1][1])
    return selected


def write_shard(rows: list[dict[str, torch.Tensor]], output: Path, offset: int) -> str:
    name = f"inverted_{offset:08d}.pt"
    payload = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
    torch.save(payload, output / name)
    return name


def main() -> None:
    args = parse_args()
    if args.stride < 1 or args.shard_size < 1:
        raise ValueError("stride and shard-size must be positive")
    source = args.source_cache.resolve()
    output = args.output_cache.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    manifest = json.loads((source / "manifest.json").read_text())
    selected = selected_sample_ids(source, manifest, args.stride)
    output.mkdir(parents=True)

    shard_names: list[str] = []
    buffer: list[dict[str, torch.Tensor]] = []
    offset = 0
    for name in manifest["shards"]:
        shard = torch.load(source / name, map_location="cpu", weights_only=True)
        keep = torch.tensor([int(item) in selected for item in shard["sample_index"].tolist()])
        if not keep.any():
            continue
        selected_shard = {key: value[keep] for key, value in shard.items()}
        for row_index in range(len(selected_shard["sample_index"])):
            buffer.append({key: value[row_index : row_index + 1] for key, value in selected_shard.items()})
            if len(buffer) == args.shard_size:
                shard_names.append(write_shard(buffer, output, offset))
                offset += len(buffer)
                buffer = []
    if buffer:
        shard_names.append(write_shard(buffer, output, offset))
        offset += len(buffer)

    if offset != len(selected):
        raise RuntimeError(f"wrote {offset} rows for {len(selected)} selected ids")
    sampling = {
        "sample_stride": args.stride,
        "terminal_window": "always retained",
        "selection": "per (split, episode): sampler windows [::stride] plus final window",
    }
    derived = dict(manifest)
    derived.update(
        {
            "parent_cache": str(source),
            "parent_samples": int(manifest["samples"]),
            "samples": offset,
            "shards": shard_names,
            "sampling": sampling,
            "derivation": "direct subset; cached conditions and inversion latents are copied from parent rows",
        }
    )
    (output / "manifest.json").write_text(json.dumps(derived, indent=2) + "\n")
    config_path = source / "cache_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config.update({"parent_cache": str(source), "derived_samples": offset, "sampling": sampling})
        (output / "cache_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"output_cache": str(output), "samples": offset, "shards": len(shard_names)}))


if __name__ == "__main__":
    main()
