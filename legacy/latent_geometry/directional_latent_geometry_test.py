"""Compare equal-length radial and tangential perturbations around z_star."""
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
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--tangent-samples", type=int, default=20)
    parser.add_argument("--radii-rms", default="0.05,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--forward-steps", type=int, default=200)
    parser.add_argument("--reverse-steps", type=int, default=200)
    parser.add_argument("--max-val-samples", type=int, default=0)
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


def normalized_executed_action(action, policy, scale):
    action = policy.normalizer["action"].unnormalize(action)
    action = action / scale.to(action.device)
    start = policy.n_obs_steps - 1
    return action[:, start : start + policy.n_action_steps]


def action_distance(first, second):
    return (first - second).square().sum(-1).mean(-1).sqrt()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def safe_ratio(numerator, denominator):
    return np.asarray(numerator) / np.maximum(np.asarray(denominator), 1e-12)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(args, data, policy, matcher, val_idx, scale, output, radii_rms):
    n = len(val_idx)
    tangent_m = args.tangent_samples
    latent_shape = tuple(data["z_star"].shape[1:])
    latent_dim = int(np.prod(latent_shape))
    sqrt_dim = math.sqrt(latent_dim)
    radii_l2 = [radius * sqrt_dim for radius in radii_rms]
    direction_count = 2 + tangent_m
    generator = torch.Generator().manual_seed(args.seed)

    center_errors = []
    zstar_norms = []
    arrays_by_radius = {
        radius: {
            key: []
            for key in (
                "out_error",
                "in_error",
                "tangent_error",
                "out_displacement",
                "in_displacement",
                "tangent_displacement",
                "out_absolute_norm",
                "in_absolute_norm",
                "tangent_absolute_norm",
            )
        }
        for radius in radii_rms
    }
    max_length_error = 0.0
    max_tangent_dot = 0.0

    for begin in range(0, n, args.batch_size):
        ids = val_idx[begin : begin + args.batch_size]
        batch = len(ids)
        condition = data["condition"][ids].to(args.device)
        expert = data["expert"][ids].to(args.device)
        z_star = data["z_star"][ids].to(args.device)
        z_flat = z_star.flatten(1)
        z_norm = z_flat.norm(dim=-1)
        radial_unit = z_flat / z_norm[:, None].clamp_min(1e-12)

        tangent = torch.randn(
            (batch, tangent_m, latent_dim), generator=generator
        ).to(args.device)
        tangent = tangent - (tangent * radial_unit[:, None]).sum(-1, keepdim=True) * radial_unit[:, None]
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        base_directions = torch.cat(
            [radial_unit[:, None], -radial_unit[:, None], tangent], dim=1
        )
        radius_tensor = torch.as_tensor(
            radii_l2, dtype=z_star.dtype, device=args.device
        )
        perturbation = radius_tensor[None, :, None, None] * base_directions[:, None]
        candidates = z_flat[:, None, None] + perturbation

        actual_length = perturbation.norm(dim=-1)
        max_length_error = max(
            max_length_error,
            float((actual_length - radius_tensor[None, :, None]).abs().max()),
        )
        max_tangent_dot = max(
            max_tangent_dot,
            float((tangent * radial_unit[:, None]).sum(-1).abs().max()),
        )

        center_raw = forward_flow(
            policy, matcher, z_star, condition, args.forward_steps, recompute=False
        )
        center_action = normalized_executed_action(center_raw, policy, scale)
        expert_action = normalized_executed_action(expert, policy, scale)
        center_error = action_distance(center_action, expert_action)
        center_errors.extend(center_error.cpu().tolist())
        zstar_norms.extend(z_norm.cpu().tolist())

        radius_count = len(radii_rms)
        candidate_raw = forward_flow(
            policy,
            matcher,
            candidates.reshape(batch * radius_count * direction_count, *latent_shape),
            condition[:, None, None]
            .expand(batch, radius_count, direction_count, -1)
            .reshape(batch * radius_count * direction_count, -1),
            args.forward_steps,
            recompute=False,
        )
        candidate_action = normalized_executed_action(candidate_raw, policy, scale)
        candidate_action = candidate_action.reshape(
            batch, radius_count, direction_count, policy.n_action_steps, policy.action_dim
        )
        target = expert_action[:, None, None].expand_as(candidate_action)
        center_target = center_action[:, None, None].expand_as(candidate_action)
        error = action_distance(
            candidate_action.flatten(0, 2), target.flatten(0, 2)
        ).reshape(batch, radius_count, direction_count)
        displacement = action_distance(
            candidate_action.flatten(0, 2), center_target.flatten(0, 2)
        ).reshape(batch, radius_count, direction_count)
        absolute_norm = candidates.norm(dim=-1)

        for radius_index, radius in enumerate(radii_rms):
            destination = arrays_by_radius[radius]
            destination["out_error"].extend(error[:, radius_index, 0].cpu().tolist())
            destination["in_error"].extend(error[:, radius_index, 1].cpu().tolist())
            destination["tangent_error"].extend(error[:, radius_index, 2:].cpu().tolist())
            destination["out_displacement"].extend(displacement[:, radius_index, 0].cpu().tolist())
            destination["in_displacement"].extend(displacement[:, radius_index, 1].cpu().tolist())
            destination["tangent_displacement"].extend(displacement[:, radius_index, 2:].cpu().tolist())
            destination["out_absolute_norm"].extend(absolute_norm[:, radius_index, 0].cpu().tolist())
            destination["in_absolute_norm"].extend(absolute_norm[:, radius_index, 1].cpu().tolist())
            destination["tangent_absolute_norm"].extend(absolute_norm[:, radius_index, 2:].cpu().tolist())

        if begin == 0 or (begin // args.batch_size) % 34 == 0 or begin + batch == n:
            last = arrays_by_radius[radii_rms[-1]]
            progress = {
                "progress": begin + batch,
                "total": n,
                "center_error": float(np.mean(center_errors)),
                "largest_radius": radii_rms[-1],
                "out_error": float(np.mean(last["out_error"])),
                "in_error": float(np.mean(last["in_error"])),
                "tangent_error": float(np.mean(last["tangent_error"])),
            }
            print(json.dumps(progress), flush=True)

    center_errors = np.asarray(center_errors, dtype=np.float32)
    zstar_norms = np.asarray(zstar_norms, dtype=np.float32)
    output_arrays = {
        "center_errors": center_errors,
        "zstar_norms": zstar_norms,
        "radii_rms": np.asarray(radii_rms, dtype=np.float32),
        "radii_l2": np.asarray(radii_l2, dtype=np.float32),
    }
    radius_report = {}
    rows = []
    for radius, radius_l2 in zip(radii_rms, radii_l2):
        source = arrays_by_radius[radius]
        values = {}
        for key, value in source.items():
            array = np.asarray(value, dtype=np.float32)
            if key.startswith("tangent_"):
                array = array.reshape(n, tangent_m)
            values[key] = array
            output_arrays[f"r{radius:g}_{key}"] = array

        tangent_condition_mean = values["tangent_error"].mean(1)
        tangent_condition_std = values["tangent_error"].std(1)
        tangent_p10 = np.quantile(values["tangent_error"], 0.10, axis=1)
        tangent_p90 = np.quantile(values["tangent_error"], 0.90, axis=1)
        tangent_spread = tangent_p90 - tangent_p10
        tangent_spread_ratio = safe_ratio(tangent_p90, tangent_p10)
        tangent_cv = safe_ratio(tangent_condition_std, tangent_condition_mean)
        tangent_gain = values["tangent_displacement"] / radius_l2
        out_gain = values["out_displacement"] / radius_l2
        in_gain = values["in_displacement"] / radius_l2
        out_minus_in = values["out_error"] - values["in_error"]
        out_minus_tangent = values["out_error"] - tangent_condition_mean
        in_minus_tangent = values["in_error"] - tangent_condition_mean

        key = f"{radius:g}"
        radius_report[key] = {
            "radius_rms": radius,
            "radius_l2": radius_l2,
            "inward_crosses_origin_fraction": float(np.mean(zstar_norms < radius_l2)),
            "absolute_latent_norm": {
                "z_star": summary(zstar_norms),
                "out": summary(values["out_absolute_norm"]),
                "in": summary(values["in_absolute_norm"]),
                "tangent": summary(values["tangent_absolute_norm"]),
            },
            "target_action_error": {
                "out": summary(values["out_error"]),
                "in": summary(values["in_error"]),
                "tangent_all_directions": summary(values["tangent_error"].ravel()),
                "tangent_condition_mean": summary(tangent_condition_mean),
            },
            "center_output_displacement": {
                "out": summary(values["out_displacement"]),
                "in": summary(values["in_displacement"]),
                "tangent_all_directions": summary(values["tangent_displacement"].ravel()),
            },
            "finite_difference_directional_gain": {
                "out": summary(out_gain),
                "in": summary(in_gain),
                "tangent_all_directions": summary(tangent_gain.ravel()),
            },
            "paired_direction_comparisons": {
                "out_minus_in_mean": float(out_minus_in.mean()),
                "out_minus_in_ci95": mean_ci95(out_minus_in),
                "fraction_out_error_gt_in": float(np.mean(out_minus_in > 0)),
                "out_minus_tangent_condition_mean": float(out_minus_tangent.mean()),
                "out_minus_tangent_ci95": mean_ci95(out_minus_tangent),
                "fraction_out_error_gt_tangent_mean": float(np.mean(out_minus_tangent > 0)),
                "in_minus_tangent_condition_mean": float(in_minus_tangent.mean()),
                "in_minus_tangent_ci95": mean_ci95(in_minus_tangent),
            },
            "tangent_directional_heterogeneity": {
                "within_condition_error_std": summary(tangent_condition_std),
                "within_condition_error_cv": summary(tangent_cv),
                "within_condition_p90_minus_p10": summary(tangent_spread),
                "within_condition_p90_over_p10": summary(tangent_spread_ratio),
                "all_direction_gain_p95_over_p05": float(
                    np.quantile(tangent_gain, 0.95)
                    / max(np.quantile(tangent_gain, 0.05), 1e-12)
                ),
            },
        }
        rows.append(
            {
                "radius_rms": radius,
                "radius_l2": radius_l2,
                "center_error_mean": center_errors.mean(),
                "out_error_mean": values["out_error"].mean(),
                "in_error_mean": values["in_error"].mean(),
                "tangent_error_mean": values["tangent_error"].mean(),
                "out_error_median": np.median(values["out_error"]),
                "in_error_median": np.median(values["in_error"]),
                "tangent_error_median": np.median(values["tangent_error"]),
                "out_minus_in_ci95_low": mean_ci95(out_minus_in)[0],
                "out_minus_in_ci95_high": mean_ci95(out_minus_in)[1],
                "fraction_out_gt_in": np.mean(out_minus_in > 0),
                "fraction_out_gt_tangent_mean": np.mean(out_minus_tangent > 0),
                "tangent_error_cv_median": np.median(tangent_cv),
                "tangent_p90_over_p10_median": np.median(tangent_spread_ratio),
                "tangent_gain_p95_over_p05": np.quantile(tangent_gain, 0.95)
                / max(np.quantile(tangent_gain, 0.05), 1e-12),
                "inward_crosses_origin_fraction": np.mean(zstar_norms < radius_l2),
            }
        )

    np.savez_compressed(output / "directional_geometry_arrays.npz", **output_arrays)
    write_rows(output / "radius_direction_summary.csv", rows)
    report = {
        "n_validation": n,
        "tangent_directions_per_condition": tangent_m,
        "latent_dimension": latent_dim,
        "radius_parameterization": "radius_rms = L2(delta_z)/sqrt(latent_dimension)",
        "directions_reused_across_radii": True,
        "center_action_error": summary(center_errors),
        "zstar_norm": summary(zstar_norms),
        "construction_checks": {
            "maximum_absolute_length_error": max_length_error,
            "maximum_absolute_tangent_dot_with_zstar_unit": max_tangent_dot,
        },
        "radii": radius_report,
    }
    return report, output_arrays


def plot_results(report, arrays, output):
    radii = arrays["radii_rms"].tolist()
    keys = [f"{radius:g}" for radius in radii]
    sqrt_dim = math.sqrt(report["latent_dimension"])
    colors = {"out": "#D33F2F", "in": "#2C65B0", "tangent": "#2A9D6F"}

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for label, field in (("radial outward", "out"), ("radial inward", "in")):
        means = [report["radii"][key]["target_action_error"][field]["mean"] for key in keys]
        ax.plot(radii, means, marker="o", linewidth=2, color=colors[field], label=label)
    tangent_mean = [
        report["radii"][key]["target_action_error"]["tangent_all_directions"]["mean"]
        for key in keys
    ]
    tangent_p05 = [
        report["radii"][key]["target_action_error"]["tangent_all_directions"]["p05"]
        for key in keys
    ]
    tangent_p95 = [
        report["radii"][key]["target_action_error"]["tangent_all_directions"]["p95"]
        for key in keys
    ]
    ax.plot(radii, tangent_mean, marker="o", linewidth=2, color=colors["tangent"], label="random tangent mean")
    ax.fill_between(radii, tangent_p05, tangent_p95, color=colors["tangent"], alpha=0.18, label="tangent 5–95%")
    ax.axhline(report["center_action_error"]["mean"], color="black", linestyle="--", linewidth=1, label=r"center $z^\star$")
    ax.set(
        xlabel=r"fixed perturbation radius $\|\Delta z\|_2/\sqrt{d}$",
        ylabel="normalized executed action error",
        yscale="log",
        title="Equal-length radial and tangential perturbations",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "directional_error_by_radius.png", dpi=220)
    fig.savefig(output / "directional_error_by_radius.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for field, label in (("out", "outward"), ("in", "inward"), ("tangent", "tangent")):
        norms = [report["radii"][key]["absolute_latent_norm"][field]["mean"] / sqrt_dim for key in keys]
        errors = [
            report["radii"][key]["target_action_error"][
                field if field != "tangent" else "tangent_all_directions"
            ]["mean"]
            for key in keys
        ]
        axes[0].plot(radii, norms, marker="o", label=label, color=colors[field])
        axes[1].plot(norms, errors, marker="o", label=label, color=colors[field])
    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1, label=r"typical $N(0,I)$ RMS norm")
    axes[0].set(xlabel=r"$\|\Delta z\|_2/\sqrt{d}$", ylabel=r"absolute latent RMS norm $\|z\|_2/\sqrt{d}$", title="Prior-norm effect")
    axes[1].set(xlabel=r"absolute latent RMS norm $\|z\|_2/\sqrt{d}$", ylabel="normalized action error", yscale="log", title="Norm versus behavioral error")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "latent_norm_direction_effect.png", dpi=220)
    fig.savefig(output / "latent_norm_direction_effect.pdf")
    plt.close(fig)

    tangent_cv = [
        report["radii"][key]["tangent_directional_heterogeneity"]["within_condition_error_cv"]["median"]
        for key in keys
    ]
    tangent_ratio = [
        report["radii"][key]["tangent_directional_heterogeneity"]["within_condition_p90_over_p10"]["median"]
        for key in keys
    ]
    gain_ratio = [
        report["radii"][key]["tangent_directional_heterogeneity"]["all_direction_gain_p95_over_p05"]
        for key in keys
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(radii, tangent_cv, marker="o", label="median within-condition error CV")
    ax.plot(radii, tangent_ratio, marker="o", label="median within-condition error p90/p10")
    ax.plot(radii, gain_ratio, marker="o", label="directional gain p95/p05")
    ax.set(
        xlabel=r"fixed perturbation radius $\|\Delta z\|_2/\sqrt{d}$",
        ylabel="directional heterogeneity",
        yscale="log",
        title=r"Tangential geometry induced by $J_F^\top J_F$",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "tangent_directional_heterogeneity.png", dpi=220)
    fig.savefig(output / "tangent_directional_heterogeneity.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    radii_rms = [float(value) for value in args.radii_rms.split(",")]
    if any(radius <= 0 for radius in radii_rms):
        raise ValueError("All radii must be positive")
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
        args, data, policy, matcher, val_idx, action_scale(data), output, radii_rms
    )
    after = digest(policy)
    report.update(
        {
            "status": "ok",
            "seed": args.seed,
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
    write_json(report, output / "directional_geometry_report.json")
    write_json({"args": vars(args), "report_file": "directional_geometry_report.json"}, output / "config.json")
    plot_results(report, arrays, output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
