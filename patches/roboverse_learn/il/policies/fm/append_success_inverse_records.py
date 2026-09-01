"""Atomically append one successful episode's captured inverse records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def append(memory_path: Path, capture_dir: Path, episode_id: int) -> dict[str, object]:
    records = [
        torch.load(path, map_location="cpu")
        for path in sorted(capture_dir.glob(f"episode_{episode_id:06d}_query_*.pt"))
    ]
    if not records:
        raise RuntimeError(f"No captured source records for successful episode {episode_id} in {capture_dir}")
    payload = torch.load(memory_path, map_location="cpu")
    values = payload["retrieval_memory"]["__global__"]
    count = len(records)
    additions = {
        "source": torch.stack([record["source"].float() for record in records]).contiguous(),
        "obs_feature": torch.stack([record["condition"].float() for record in records]).contiguous(),
        "raw_action": torch.stack([record["raw_action"].float() for record in records]).contiguous(),
        # Deliberately outside demo-ID range: this is past rollout experience,
        # never an expert-demo self match.
        "episode_index": torch.full((count,), 1_000_000 + episode_id, dtype=torch.long),
        "global_start": torch.tensor(
            # `global_start` is a trajectory coordinate.  It must advance by
            # exactly the executed action chunk, not by one query counter, so
            # temporal retrieval can recover real predecessor/successor paths
            # for online experience just as it does for expert windows.
            [
                1_000_000_000 + episode_id * 100_000
                + int(record["query_id"]) * int(record.get("chunk_stride", 8))
             for record in records], dtype=torch.long,
        ),
        "purity": torch.ones(count, dtype=torch.float32),
    }
    if all("proprio_feature" in record for record in records):
        additions["proprio_feature"] = torch.stack([
            record["proprio_feature"].float() for record in records
        ]).contiguous()
    if all("proprio_state" in record for record in records):
        additions["proprio_state"] = torch.stack([
            record["proprio_state"].float() for record in records
        ]).contiguous()
    for key in ("source", "obs_feature", "raw_action"):
        if tuple(additions[key].shape[1:]) != tuple(values[key].shape[1:]):
            raise RuntimeError(
                f"{key} shape mismatch: {tuple(additions[key].shape)} vs {tuple(values[key].shape)}"
            )
    original_count = len(values["source"])
    for key, tensor in list(values.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0 or tensor.shape[0] != original_count:
            continue
        addition = additions.get(key)
        if addition is None:
            # DINO is intentionally absent for online records.  In contrast,
            # Proprio entries must be real whenever temporal retrieval is
            # enabled and are provided above from the policy-normalized query.
            addition = torch.zeros((count, *tensor.shape[1:]), dtype=tensor.dtype)
        values[key] = torch.cat([tensor.cpu(), addition.to(dtype=tensor.dtype)], dim=0).contiguous()
    meta = payload.setdefault("retrieval_memory_meta", {})
    meta.update({
        "online_success_memory": True,
        "online_memory_reader": "FlowConditionQKVRunner only",
        "online_record_semantics": "past whole-episode-success condition-source association",
    })
    temporary = memory_path.with_suffix(memory_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(memory_path)
    return {
        "episode_id": episode_id,
        "inserted_records": count,
        "bank_items_before": original_count,
        "bank_items_after": int(values["source"].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-path", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--episode-id", required=True, type=int)
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()
    result = append(Path(args.memory_path), Path(args.capture_dir), args.episode_id)
    if args.summary_path:
        Path(args.summary_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
