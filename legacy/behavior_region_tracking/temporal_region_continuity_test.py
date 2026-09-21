"""Test whether an executed behavior region is continuous across policy updates.

For adjacent samples from the same episode, the previous inversion latent is
decoded under the current condition:

    E_reuse = d_A(F(z_{t-1}^* | c_t), A_t^*)

This is an offline continuity diagnostic.  The previous expert inversion
latent is never used as a train-bank entry for observation retrieval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from behavior_region_localizer import (
    action_scale,
    build_bank,
    check_settings,
    decode_direct_errors,
    decode_pair_errors,
    digest,
    load_cache,
    load_flow,
    seed_all,
    sha256,
    write_json,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--forward-steps", type=int, default=200)
    parser.add_argument("--global-samples", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "both"), default="val")
    return parser.parse_args()


def make_adjacent_pairs(data: dict, split_value: int) -> torch.Tensor:
    """Return [previous_row,current_row] for adjacent physical timesteps."""
    rows = []
    for episode in torch.unique(data["episode"]):
        episode_rows = torch.where(
            (data["episode"] == episode) & (data["split"] == split_value)
        )[0]
        if len(episode_rows) < 2:
            continue
        order = torch.argsort(data["condition_id"][episode_rows])
        ordered = episode_rows[order]
        previous = ordered[:-1]
        current = ordered[1:]
        adjacent = (
            (data["condition_id"][current] - data["condition_id"][previous] == 1)
            & (data["sampler_index"][current] - data["sampler_index"][previous] == 1)
        )
        rows.extend(torch.stack([previous[adjacent], current[adjacent]], dim=1).tolist())
    if not rows:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def paired_ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    half = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    return [float(values.mean() - half), float(values.mean() + half)]


def summarize(rows: list[dict], split: str) -> dict:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        return {"n_pairs": 0}
    current = np.asarray([row["e_current"] for row in selected], dtype=np.float64)
    reuse = np.asarray([row["e_reuse"] for row in selected], dtype=np.float64)
    gaussian = np.asarray([row["e_gaussian"] for row in selected], dtype=np.float64)
    retrieval = np.asarray([row["e_obs_retrieval"] for row in selected], dtype=np.float64)
    latent_distance = np.asarray([row["latent_distance_rms"] for row in selected], dtype=np.float64)
    return {
        "n_pairs": len(selected),
        "e_current_mean": float(current.mean()),
        "e_current_median": float(np.median(current)),
        "e_reuse_mean": float(reuse.mean()),
        "e_reuse_median": float(np.median(reuse)),
        "e_reuse_ci95": paired_ci(reuse),
        "e_gaussian_mean": float(gaussian.mean()),
        "e_obs_retrieval_mean": float(retrieval.mean()),
        "latent_distance_rms_mean": float(latent_distance.mean()),
        "latent_distance_rms_median": float(np.median(latent_distance)),
        "reuse_over_gaussian": float(reuse.mean() / max(gaussian.mean(), 1e-12)),
        "reuse_over_observation_retrieval": float(reuse.mean() / max(retrieval.mean(), 1e-12)),
        "fraction_reuse_better_than_gaussian": float(np.mean(reuse < gaussian)),
        "fraction_reuse_better_than_observation_retrieval": float(np.mean(reuse < retrieval)),
        "fraction_current_better_than_reuse": float(np.mean(current < reuse)),
        "fraction_reuse_region_like": float(np.mean(reuse <= np.quantile(reuse, .95))),
    }


@torch.no_grad()
def evaluate_pairs(
    args: argparse.Namespace,
    data: dict,
    bank: dict,
    policy: nn.Module,
    matcher,
    pairs: torch.Tensor,
    scale: torch.Tensor,
    device: str,
) -> list[dict]:
    previous = pairs[:, 0]
    current = pairs[:, 1]
    z_current = data["z_star"][current, None].float()
    z_previous = data["z_star"][previous, None].float()
    e_current = decode_direct_errors(
        policy, matcher, data, current, z_current, scale,
        args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
    )[:, 0]
    e_reuse = decode_direct_errors(
        policy, matcher, data, current, z_previous, scale,
        args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
    )[:, 0]

    generator = torch.Generator().manual_seed(args.seed + 101)
    z_gaussian = torch.randn(
        (len(current), args.global_samples, *data["z_star"].shape[1:]),
        generator=generator,
    )
    gaussian_errors = decode_direct_errors(
        policy, matcher, data, current, z_gaussian, scale,
        args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    e_gaussian = gaussian_errors.mean(1)

    # Observation retrieval is a train-bank-only baseline.  Its input is the
    # current causal context; the previous validation latent is not searched.
    bank_feature = F.normalize(bank["condition_feature"].float(), dim=-1)
    current_feature = F.normalize(data["context"][current].flatten(1).float(), dim=-1)
    nearest = (current_feature @ bank_feature.T).argmax(-1)
    retrieval_errors = decode_pair_errors(
        policy, matcher, data, bank, current, nearest[:, None], scale,
        args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
    )[:, 0]

    latent_distance = (
        data["z_star"][current].flatten(1) - data["z_star"][previous].flatten(1)
    ).norm(dim=-1) / math.sqrt(int(np.prod(data["z_star"].shape[1:])))
    rows = []
    for i, (previous_id, current_id) in enumerate(pairs.tolist()):
        rows.append({
            "split": "train" if int(data["split"][current_id]) == 0 else "val",
            "episode_id": int(data["episode"][current_id]),
            "previous_sample_index": int(previous_id),
            "current_sample_index": int(current_id),
            "previous_timestep": int(data["condition_id"][previous_id]),
            "current_timestep": int(data["condition_id"][current_id]),
            "latent_distance_rms": float(latent_distance[i]),
            "e_current": float(e_current[i]),
            "e_reuse": float(e_reuse[i]),
            "e_gaussian": float(e_gaussian[i]),
            "e_gaussian_best": float(gaussian_errors[i].min()),
            "e_obs_retrieval": float(retrieval_errors[i]),
            "obs_retrieval_bank_index": int(nearest[i]),
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.forward_steps != 200:
        raise ValueError("formal continuity evaluation requires 200-step Flow")
    seed_all(args.seed)
    device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache)
    policy, matcher, _ = load_flow(args)
    check_settings(data, manifest, policy, args)
    before = digest(policy)
    bank = build_bank(data, output)
    requested = (0, 1) if args.split == "both" else ((0,) if args.split == "train" else (1,))
    pairs = torch.cat([make_adjacent_pairs(data, split) for split in requested], dim=0)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    if len(pairs) == 0:
        raise ValueError("no adjacent same-episode pairs found")
    rows = evaluate_pairs(args, data, bank, policy, matcher, pairs, action_scale(data), device)
    write_rows(output / "temporal_continuity_pairs.csv", rows)
    report = {
        "status": "ok",
        "split": args.split,
        "n_pairs": len(rows),
        "metrics": {name: summarize(rows, name) for name in ("train", "val")},
        "interpretation": {
            "continuity_support": "reuse remains below Gaussian and observation retrieval while current is lower than reuse",
            "discontinuity_support": "reuse rises toward Gaussian after a behavior/environment change",
            "note": "this cache contains normal expert episodes only; cube-shift change points require closed-loop rollout logs",
        },
        "flow_steps": args.forward_steps,
        "global_samples": args.global_samples,
        "flow_checkpoint_sha256": sha256(args.checkpoint),
        "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"),
        "flow_parameter_digest_before": before,
        "flow_parameter_digest_after": digest(policy),
        "flow_unchanged": before == digest(policy),
        "all_flow_parameter_grads_none": all(p.grad is None for p in policy.parameters()),
        "leakage_checks": {
            "bank_split": "train_only",
            "previous_latent": "offline continuity diagnostic only",
            "current_validation_expert_action": "used only to compute errors",
            "future_observation": False,
            "pairing": "same-episode adjacent condition_id and sampler_index only",
        },
    }
    write_json(report, output / "temporal_continuity_report.json")
    write_json({"args": vars(args), "report_file": "temporal_continuity_report.json"}, output / "config.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
