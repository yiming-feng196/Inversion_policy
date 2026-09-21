"""Evaluate BRL geometry-constrained initialization on held-out conditions.

This is Stage 2 of the experiment.  The anchor is the train-bank latent
retrieved by Stage 1; validation expert latents are never used as an anchor.
The validation expert action is used only to compute the offline diagnostic
error.
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

from behavior_region_localizer import (
    action_scale,
    check_settings,
    decode_direct_errors,
    digest,
    executed_action_error,
    load_cache,
    load_flow,
    seed_all,
    sha256,
    write_json,
    write_rows,
)
from flow_latent_predictor_common import forward_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--flow-batch-size", type=int, default=16)
    parser.add_argument("--local-samples", type=int, default=8)
    parser.add_argument("--sigmas", default="0.05,0.10,0.25")
    parser.add_argument("--forward-steps", type=int, default=200)
    parser.add_argument("--epsilon-region", type=float, default=-1.0)
    return parser.parse_args()


def load_stage1(stage1_dir: Path) -> tuple[dict, dict, float]:
    bank = torch.load(stage1_dir / "bank.pt", map_location="cpu", weights_only=False)
    retrieval = torch.load(stage1_dir / "retrieval_results.pt", map_location="cpu", weights_only=False)
    threshold = float(retrieval.get("epsilon_region", -1.0))
    if threshold < 0:
        threshold_file = stage1_dir / "region_threshold.json"
        if threshold_file.exists():
            threshold = float(json.loads(threshold_file.read_text())["epsilon_region"])
    if threshold < 0:
        raise ValueError("Stage 1 did not save epsilon_region")
    return bank, retrieval, threshold


def norm_preserving_tangent(
    anchor: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample a tangent perturbation and project back to anchor norm."""
    flat = anchor.flatten(1)
    random = torch.randn(flat.shape, generator=generator, dtype=anchor.dtype)
    radial = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = random - (random * radial).sum(-1, keepdim=True) * radial
    tangent = tangent / tangent.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    candidate = flat + sigma * tangent
    candidate = candidate * (flat.norm(dim=-1, keepdim=True) / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    return candidate.reshape_as(anchor)


def tangent_sample(anchor: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    flat = anchor.flatten(1)
    random = torch.randn(flat.shape, generator=generator, dtype=anchor.dtype)
    radial = flat / flat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tangent = random - (random * radial).sum(-1, keepdim=True) * radial
    tangent = tangent / tangent.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return (flat + sigma * tangent).reshape_as(anchor)


def isotropic_sample(anchor: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    return anchor + sigma * torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)


@torch.no_grad()
def evaluate_variant(
    policy: nn.Module,
    matcher,
    data: dict,
    query_ids: torch.Tensor,
    z: torch.Tensor,
    anchor: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    errors = decode_direct_errors(
        policy, matcher, data, query_ids, z, scale,
        args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    z_flat = z.flatten(2)
    anchor_flat = anchor[:, None].flatten(2)
    support_distance = (z_flat - anchor_flat).norm(dim=-1).cpu().numpy() / math.sqrt(z_flat.shape[-1])
    latent_norm = z_flat.norm(dim=-1).cpu().numpy()
    anchor_norm = anchor_flat.norm(dim=-1).cpu().numpy()
    norm_change = latent_norm - anchor_norm
    violation = errors > epsilon
    return errors, support_distance, norm_change, violation


def main() -> None:
    args = parse_args()
    if args.forward_steps != 200:
        raise ValueError("formal Stage 2 evaluation must use 200-step Flow")
    seed_all(args.seed)
    device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache)
    policy, matcher, _ = load_flow(args)
    check_settings(data, manifest, policy, args)
    before = digest(policy)
    bank, retrieval, saved_epsilon = load_stage1(Path(args.stage1_dir))
    epsilon = saved_epsilon if args.epsilon_region < 0 else args.epsilon_region
    query_ids = retrieval["val_indices"].long()
    if torch.any(data["split"][query_ids] != 1):
        raise AssertionError("Stage 2 queries must be validation samples")
    anchor_ids = retrieval["brl_top_indices"][:, 0].long()
    anchor = bank["z_star"][anchor_ids].float()
    scale = action_scale(data)
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    generator = torch.Generator().manual_seed(args.seed + 71)
    rows = []
    summaries = {}

    variants = ("anchor", "isotropic", "tangent", "tangent_norm")
    for sigma in sigmas:
        samples = {}
        if sigma == 0:
            samples["anchor"] = anchor[:, None]
            continue
        samples["isotropic"] = torch.stack(
            [isotropic_sample(anchor, sigma, generator) for _ in range(args.local_samples)], dim=1
        )
        samples["tangent"] = torch.stack(
            [tangent_sample(anchor, sigma, generator) for _ in range(args.local_samples)], dim=1
        )
        samples["tangent_norm"] = torch.stack(
            [norm_preserving_tangent(anchor, sigma, generator) for _ in range(args.local_samples)], dim=1
        )
        for variant, z in samples.items():
            errors, support, norm_change, violation = evaluate_variant(
                policy, matcher, data, query_ids, z, anchor, scale, epsilon, args, device,
            )
            key = f"{variant}_sigma_{sigma:g}"
            summaries[key] = {
                "sigma": sigma,
                "variant": variant,
                "mean": float(errors.mean()),
                "median": float(np.median(errors)),
                "p90": float(np.quantile(errors, .90)),
                "p95": float(np.quantile(errors, .95)),
                "support_distance_rms": float(support.mean()),
                "absolute_norm_change": float(np.abs(norm_change).mean()),
                "signed_norm_change": float(norm_change.mean()),
                "region_violation_rate": float(violation.mean()),
            }
            for i, qid in enumerate(query_ids.tolist()):
                for j in range(z.shape[1]):
                    rows.append({
                        "sample_index": int(qid),
                        "episode_id": int(data["episode"][qid]),
                        "sigma": sigma,
                        "variant": variant,
                        "sample": j,
                        "action_error": float(errors[i, j]),
                        "support_distance_rms": float(support[i, j]),
                        "signed_norm_change": float(norm_change[i, j]),
                        "region_violation": int(violation[i, j]),
                    })

    # Anchor is evaluated once, independent of sigma.
    anchor_z = anchor[:, None]
    errors, support, norm_change, violation = evaluate_variant(
        policy, matcher, data, query_ids, anchor_z, anchor, scale, epsilon, args, device,
    )
    summaries["anchor"] = {
        "sigma": 0.0,
        "variant": "anchor",
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, .90)),
        "p95": float(np.quantile(errors, .95)),
        "support_distance_rms": float(support.mean()),
        "absolute_norm_change": float(np.abs(norm_change).mean()),
        "signed_norm_change": float(norm_change.mean()),
        "region_violation_rate": float(violation.mean()),
    }
    for i, qid in enumerate(query_ids.tolist()):
        rows.append({
            "sample_index": int(qid),
            "episode_id": int(data["episode"][qid]),
            "sigma": 0.0,
            "variant": "anchor",
            "sample": 0,
            "action_error": float(errors[i, 0]),
            "support_distance_rms": float(support[i, 0]),
            "signed_norm_change": float(norm_change[i, 0]),
            "region_violation": int(violation[i, 0]),
        })

    write_rows(output / "geometry_samples.csv", rows)
    write_json({
        "epsilon_region": epsilon,
        "variants": summaries,
        "anchor_source": "train_only_bank_top1",
        "validation_expert_action_used_only_for_error": True,
        "flow_steps": args.forward_steps,
        "flow_checkpoint_sha256": sha256(args.checkpoint),
        "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"),
        "flow_parameter_digest_before": before,
        "flow_parameter_digest_after": digest(policy),
        "flow_unchanged": before == digest(policy),
        "all_flow_parameter_grads_none": all(p.grad is None for p in policy.parameters()),
    }, output / "stage2_report.json")
    write_json({"args": vars(args), "report_file": "stage2_report.json"}, output / "config.json")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
