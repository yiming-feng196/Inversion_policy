"""Test whether behavior-specific latents occupy a local region around z_star."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_latent_predictor_common import (
    forward_flow,
    load_cache,
    load_flow,
    seed_all,
    sha256,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--local-samples", type=int, default=20)
    parser.add_argument("--global-samples", type=int, default=100)
    parser.add_argument("--sigmas", default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--forward-steps", type=int, default=200)
    parser.add_argument("--reverse-steps", type=int, default=200)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--scatter-points-per-group", type=int, default=12000)
    return parser.parse_args()


def digest(module):
    h = hashlib.sha256()
    for name, parameter in module.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_settings(data, manifest, policy, args):
    if args.forward_steps != 200 or args.reverse_steps != 200:
        raise ValueError("This test requires 200-step Flow evaluation")
    if manifest["forward_steps"] != 200 or manifest["reverse_steps"] != 200:
        raise ValueError("Cache solver settings are not 200/200")
    if tuple(data["context"].shape[1:]) != (12, 521):
        raise ValueError("Expected verified 12-frame context")
    if tuple(data["z_star"].shape[1:]) != (16, 9):
        raise ValueError("Expected native [16, 9] latent shape")
    if tuple(data["expert"].shape[1:]) != (16, 9):
        raise ValueError("Expected native [16, 9] action shape")
    if policy.horizon != 16 or policy.action_dim != 9:
        raise ValueError("Checkpoint must have horizon=16 and action_dim=9")
    if set(manifest["train_episodes"]) & set(manifest["val_episodes"]):
        raise AssertionError("Train/validation episodes overlap")
    if not torch.all(data["observation_indices"] <= data["condition_id"][:, None]):
        raise AssertionError("Future observation leakage")
    if not torch.equal(data["observation_indices"][:, -1], data["condition_id"]):
        raise AssertionError("Context does not end at current timestep")
    if not torch.equal(data["condition"], data["context"][:, -8:].flatten(1)):
        raise AssertionError("Flow condition is not final eight context frames")
    if not all(not p.requires_grad and p.grad is None for p in policy.parameters()):
        raise AssertionError("Flow is not frozen")


def action_scale(data):
    train = data["expert_raw"][data["split"] == 0]
    return train.flatten(0, 1).std(0, unbiased=False).clamp_min(1e-6)


def executed_action_error(pred, target, policy, scale):
    pred = policy.normalizer["action"].unnormalize(pred)
    target = policy.normalizer["action"].unnormalize(target)
    delta = (pred - target) / scale.to(pred.device)
    start = policy.n_obs_steps - 1
    delta = delta[:, start : start + policy.n_action_steps]
    return delta.square().sum(-1).mean(-1).sqrt()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args, data, policy, matcher, val_idx, scale, output, sigmas):
    n = len(val_idx)
    local_m = args.local_samples
    global_m = args.global_samples
    latent_dim = int(np.prod(data["z_star"].shape[1:]))
    generator = torch.Generator().manual_seed(args.seed)

    center_errors = []
    local_distances = {sigma: [] for sigma in sigmas}
    local_errors = {sigma: [] for sigma in sigmas}
    local_condition_means = {sigma: [] for sigma in sigmas}
    global_distances = []
    global_errors = []
    global_condition_means = []
    rows = []

    for begin in range(0, n, args.batch_size):
        ids = val_idx[begin : begin + args.batch_size]
        batch = len(ids)
        condition = data["condition"][ids].to(args.device)
        expert = data["expert"][ids].to(args.device)
        z_star = data["z_star"][ids].to(args.device)

        center_action = forward_flow(
            policy, matcher, z_star, condition, args.forward_steps, recompute=False
        )
        center = executed_action_error(center_action, expert, policy, scale)
        center_errors.extend(center.cpu().tolist())

        # Reuse directions across radii so sigma is the only changing variable.
        epsilon = torch.randn(
            (batch, local_m, 16, 9), generator=generator
        ).to(args.device)
        sigma_tensor = torch.as_tensor(sigmas, device=args.device, dtype=z_star.dtype)
        z_local = (
            z_star[:, None, None]
            + sigma_tensor[None, :, None, None, None] * epsilon[:, None]
        )
        sigma_count = len(sigmas)
        local_action = forward_flow(
            policy,
            matcher,
            z_local.reshape(batch * sigma_count * local_m, 16, 9),
            condition[:, None, None]
            .expand(batch, sigma_count, local_m, -1)
            .reshape(batch * sigma_count * local_m, -1),
            args.forward_steps,
            recompute=False,
        )
        local_error = executed_action_error(
            local_action,
            expert[:, None, None]
            .expand(batch, sigma_count, local_m, 16, 9)
            .reshape(batch * sigma_count * local_m, 16, 9),
            policy,
            scale,
        ).reshape(batch, sigma_count, local_m)
        local_distance = (z_local - z_star[:, None, None]).flatten(3).norm(dim=-1)
        for sigma_index, sigma in enumerate(sigmas):
            error = local_error[:, sigma_index]
            distance = local_distance[:, sigma_index]
            distance_rms = distance / math.sqrt(latent_dim)
            local_errors[sigma].extend(error.cpu().flatten().tolist())
            local_distances[sigma].extend(distance_rms.cpu().flatten().tolist())
            local_condition_means[sigma].extend(error.mean(-1).cpu().tolist())

        z_global = torch.randn((batch, global_m, 16, 9), generator=generator).to(
            args.device
        )
        global_action = forward_flow(
            policy,
            matcher,
            z_global.reshape(batch * global_m, 16, 9),
            condition[:, None].expand(batch, global_m, -1).reshape(batch * global_m, -1),
            args.forward_steps,
            recompute=False,
        )
        global_error = executed_action_error(
            global_action,
            expert[:, None].expand(batch, global_m, 16, 9).reshape(batch * global_m, 16, 9),
            policy,
            scale,
        ).reshape(batch, global_m)
        global_distance = (z_global - z_star[:, None]).flatten(2).norm(dim=-1)
        global_distance_rms = global_distance / math.sqrt(latent_dim)
        global_errors.extend(global_error.cpu().flatten().tolist())
        global_distances.extend(global_distance_rms.cpu().flatten().tolist())
        global_condition_means.extend(global_error.mean(-1).cpu().tolist())

        for offset, sample_id in enumerate(ids.tolist()):
            row = {
                "sample_index": sample_id,
                "episode_id": int(data["episode"][sample_id]),
                "center_error": float(center[offset]),
                "global_mean_error": float(global_error[offset].mean()),
                "global_median_error": float(global_error[offset].median()),
            }
            for sigma in sigmas:
                start = (begin + offset) * local_m
                stop = start + local_m
                values = local_errors[sigma][start:stop]
                row[f"local_sigma_{sigma:g}_mean_error"] = float(np.mean(values))
                row[f"local_sigma_{sigma:g}_median_error"] = float(np.median(values))
            rows.append(row)

        if begin == 0 or (begin // args.batch_size) % 25 == 0 or begin + batch == n:
            progress = {
                "progress": begin + batch,
                "total": n,
                "center_mean": float(np.mean(center_errors)),
                "local_means": {
                    str(s): float(np.mean(local_errors[s])) for s in sigmas
                },
                "global_mean": float(np.mean(global_errors)),
            }
            print(json.dumps(progress), flush=True)

    center_errors = np.asarray(center_errors, dtype=np.float32)
    global_errors = np.asarray(global_errors, dtype=np.float32)
    global_distances = np.asarray(global_distances, dtype=np.float32)
    global_condition_means = np.asarray(global_condition_means, dtype=np.float32)
    arrays = {
        "center_errors": center_errors,
        "global_errors": global_errors,
        "global_distance_rms": global_distances,
        "global_condition_means": global_condition_means,
        "sigmas": np.asarray(sigmas, dtype=np.float32),
    }
    global_stats = {
        "distance_rms": summary(global_distances),
        "action_error": summary(global_errors),
        "condition_mean_action_error": summary(global_condition_means),
    }
    local_stats = {}
    paired_rows = []
    for sigma in sigmas:
        distances = np.asarray(local_distances[sigma], dtype=np.float32)
        errors = np.asarray(local_errors[sigma], dtype=np.float32)
        condition_means = np.asarray(local_condition_means[sigma], dtype=np.float32)
        difference = condition_means - global_condition_means
        ratio = condition_means / np.maximum(global_condition_means, 1e-12)
        key = f"{sigma:g}"
        arrays[f"local_sigma_{key}_distance_rms"] = distances
        arrays[f"local_sigma_{key}_errors"] = errors
        arrays[f"local_sigma_{key}_condition_means"] = condition_means
        local_stats[key] = {
            "distance_rms": summary(distances),
            "action_error": summary(errors),
            "condition_mean_action_error": summary(condition_means),
            "paired_local_minus_global_mean": float(difference.mean()),
            "paired_difference_ci95": mean_ci95(difference),
            "condition_mean_local_over_global": summary(ratio),
            "fraction_conditions_local_better": float(np.mean(difference < 0)),
            "fraction_points_below_global_median": float(
                np.mean(errors < np.median(global_errors))
            ),
        }
        ci = mean_ci95(difference)
        paired_rows.append(
            {
                "sigma": sigma,
                "distance_rms_mean": distances.mean(),
                "action_error_mean": errors.mean(),
                "action_error_median": np.median(errors),
                "action_error_p95": np.quantile(errors, 0.95),
                "global_action_error_mean": global_errors.mean(),
                "local_over_global_mean": errors.mean() / global_errors.mean(),
                "paired_difference_ci95_low": ci[0],
                "paired_difference_ci95_high": ci[1],
                "fraction_conditions_local_better": np.mean(difference < 0),
            }
        )

    np.savez_compressed(output / "local_region_arrays.npz", **arrays)
    write_rows(output / "sample_level_summary.csv", rows)
    write_rows(output / "radius_summary.csv", paired_rows)
    report = {
        "n_validation": n,
        "local_samples_per_sigma_per_condition": local_m,
        "global_samples_per_condition": global_m,
        "latent_dimension": latent_dim,
        "distance_definition": "L2(z-z_star)/sqrt(latent_dimension)",
        "same_epsilon_directions_across_sigmas": True,
        "center_action_error": summary(center_errors),
        "global_gaussian": global_stats,
        "local": local_stats,
        "criterion": (
            "A radius supports a behavior-specific region when its paired "
            "local-minus-global action-error CI95 is entirely below zero."
        ),
    }
    return report, arrays


def plot_results(report, arrays, output, max_points, seed):
    rng = np.random.default_rng(seed)
    sigmas = arrays["sigmas"].tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    global_distance = arrays["global_distance_rms"]
    global_error = arrays["global_errors"]
    take = min(max_points, len(global_error))
    idx = rng.choice(len(global_error), take, replace=False)
    ax.scatter(
        global_distance[idx],
        global_error[idx],
        s=6,
        alpha=0.10,
        color="0.45",
        edgecolors="none",
        label="global N(0, I)",
        rasterized=True,
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(sigmas)))
    for sigma, color in zip(sigmas, colors):
        key = f"{sigma:g}"
        distance = arrays[f"local_sigma_{key}_distance_rms"]
        error = arrays[f"local_sigma_{key}_errors"]
        take = min(max_points, len(error))
        idx = rng.choice(len(error), take, replace=False)
        ax.scatter(
            distance[idx],
            error[idx],
            s=6,
            alpha=0.12,
            color=color,
            edgecolors="none",
            label=fr"local $\sigma={sigma:g}$",
            rasterized=True,
        )
    ax.set(
        xlabel=r"distance from $z^\star$: $\|z-z^\star\|_2/\sqrt{d}$",
        ylabel="normalized executed action error",
        yscale="log",
        title="Behavior-specific latent region around $z^\star$",
    )
    ax.grid(alpha=0.2)
    ax.legend(markerscale=2, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "distance_vs_action_error.png", dpi=220)
    fig.savefig(output / "distance_vs_action_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = [report["local"][f"{sigma:g}"]["distance_rms"]["mean"] for sigma in sigmas]
    median = [report["local"][f"{sigma:g}"]["action_error"]["median"] for sigma in sigmas]
    p05 = [report["local"][f"{sigma:g}"]["action_error"]["p05"] for sigma in sigmas]
    p95 = [report["local"][f"{sigma:g}"]["action_error"]["p95"] for sigma in sigmas]
    ax.plot(x, median, marker="o", color="#2457C5", label="local median")
    ax.fill_between(x, p05, p95, alpha=0.2, color="#2457C5", label="local 5–95%")
    ax.scatter(
        [report["global_gaussian"]["distance_rms"]["mean"]],
        [report["global_gaussian"]["action_error"]["median"]],
        marker="*",
        s=180,
        color="#C83E31",
        label="global N(0, I) median",
        zorder=5,
    )
    ax.axhline(
        report["center_action_error"]["median"],
        linestyle="--",
        color="black",
        linewidth=1,
        label=r"center $z^\star$ median",
    )
    ax.set(
        xlabel=r"distance from $z^\star$: $\|z-z^\star\|_2/\sqrt{d}$",
        ylabel="normalized executed action error",
        yscale="log",
        title="Local radius–error profile",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "radius_error_profile.png", dpi=220)
    fig.savefig(output / "radius_error_profile.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    sigmas = [float(value) for value in args.sigmas.split(",")]
    if any(value <= 0 for value in sigmas):
        raise ValueError("All sigmas must be positive")
    seed_all(args.seed)
    args.device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache)
    policy, matcher, _ = load_flow(args)
    check_settings(data, manifest, policy, args)
    before = digest(policy)
    val_idx = torch.where(data["split"] == 1)[0]
    if args.max_val_samples:
        val_idx = val_idx[: args.max_val_samples]
    report, arrays = evaluate(
        args, data, policy, matcher, val_idx, action_scale(data), output, sigmas
    )
    after = digest(policy)
    report.update(
        {
            "status": "ok",
            "seed": args.seed,
            "sigmas": sigmas,
            "forward_steps": args.forward_steps,
            "reverse_steps": args.reverse_steps,
            "shape": [16, 9],
            "flow_checkpoint_sha256": sha256(args.checkpoint),
            "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"),
            "flow_parameter_digest_before": before,
            "flow_parameter_digest_after": after,
            "flow_unchanged": before == after,
            "all_flow_parameter_grads_none": all(p.grad is None for p in policy.parameters()),
        }
    )
    write_json(report, output / "local_region_report.json")
    write_json({"args": vars(args), "report_file": "local_region_report.json"}, output / "config.json")
    plot_results(report, arrays, output, args.scatter_points_per_group, args.seed)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
