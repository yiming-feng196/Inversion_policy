"""Make global PCA and local frozen-Flow behavioral-geometry figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ModuleNotFoundError:
    plt = None
    LogNorm = None

from flow_latent_predictor_common import (
    forward_flow,
    load_cache,
    load_flow,
    seed_all,
    sha256,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--global-npz", required=True)
    p.add_argument("--directional-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--forward-steps", type=int, default=200)
    p.add_argument("--grid-size", type=int, default=61)
    p.add_argument("--grid-batch-size", type=int, default=128)
    p.add_argument("--terrain-radius-rms", type=float, default=0.75)
    p.add_argument("--anisotropy-radius-rms", type=float, default=0.25)
    return p.parse_args()


def digest(module):
    h = hashlib.sha256()
    for name, parameter in module.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def action_scale(data):
    train = data["expert_raw"][data["split"] == 0]
    return train.flatten(0, 1).std(0, unbiased=False).clamp_min(1e-6)


def normalized_executed_action(action, policy, scale):
    action = policy.normalizer["action"].unnormalize(action)
    action = action / scale.to(action.device)
    start = policy.n_obs_steps - 1
    return action[:, start : start + policy.n_action_steps]


def action_vector(action, policy, scale):
    """Vector whose Euclidean distance equals the experiment action metric."""
    return normalized_executed_action(action, policy, scale).flatten(1) / math.sqrt(
        policy.n_action_steps
    )


def robust_distance_from_median(x):
    x = np.asarray(x, dtype=np.float64)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return np.abs(x - median) / max(mad, 1e-12)


def select_representative(direction_data):
    center = direction_data["center_errors"]
    norms = direction_data["zstar_norms"]
    radius = 0.1
    tangent_error = direction_data[f"r{radius:g}_tangent_error"]
    tangent_displacement = direction_data[f"r{radius:g}_tangent_displacement"]
    gains = tangent_displacement / (radius * math.sqrt(144))
    anisotropy = np.quantile(gains, 0.90, axis=1) / np.maximum(
        np.quantile(gains, 0.10, axis=1), 1e-12
    )
    score = (
        robust_distance_from_median(center)
        + robust_distance_from_median(norms)
        + robust_distance_from_median(anisotropy)
    )
    ordinal = int(np.argmin(score))
    return ordinal, {
        "selection_rule": (
            "minimum sum of absolute MAD-scaled distances from validation medians "
            "for center action error, z_star L2 norm, and r_rms=0.1 tangent p90/p10 gain"
        ),
        "validation_ordinal": ordinal,
        "selection_score": float(score[ordinal]),
        "center_error": float(center[ordinal]),
        "z_star_l2_norm": float(norms[ordinal]),
        "tangent_gain_p90_p10": float(anisotropy[ordinal]),
        "population_medians": {
            "center_error": float(np.median(center)),
            "z_star_l2_norm": float(np.median(norms)),
            "tangent_gain_p90_p10": float(np.median(anisotropy)),
        },
    }


def pca_fit_transform(arrays):
    merged = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = merged.mean(axis=0)
    centered = merged - mean
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[:3]]
    transformed = [(x - mean) @ components for x in arrays]
    explained = eigenvalues[:3] / eigenvalues.sum()
    return transformed, explained, mean, components, eigenvalues


def make_pca_figures(global_npz, output, seed):
    source = np.load(global_npz)
    names = ["native", "cycle", "expert"]
    arrays = [source[name].reshape(len(source[name]), -1) for name in names]
    projected, explained, mean, components, eigenvalues = pca_fit_transform(arrays)
    colors = ["#3274A1", "#E1812C", "#3A923A"]
    rng = np.random.default_rng(seed)
    sample_n = min(3500, *(len(x) for x in projected))
    sampled = [x[rng.choice(len(x), sample_n, replace=False)] for x in projected]

    np.savez_compressed(
        output / "global_latent_pca_arrays.npz",
        native=projected[0],
        cycle=projected[1],
        expert=projected[2],
        explained_variance_ratio=explained,
        pca_mean=mean,
        pca_components=components.T,
        eigenvalues=eigenvalues,
    )
    if plt is not None:
        fig = plt.figure(figsize=(13.2, 5.4), constrained_layout=True)
        ax2 = fig.add_subplot(1, 2, 1)
        for name, points, color in zip(names, sampled, colors):
            ax2.scatter(points[:, 0], points[:, 1], s=7, alpha=0.22, c=color, label=name)
        ax2.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
        ax2.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
        ax2.set_title("Global latent distributions: PCA 2D")
        ax2.legend(frameon=False, markerscale=2.0)
        ax2.grid(alpha=0.16)

        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
        for name, points, color in zip(names, sampled, colors):
            ax3.scatter(
                points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.16, c=color, label=name
            )
        ax3.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
        ax3.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
        ax3.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)")
        ax3.set_title("Global latent distributions: PCA 3D")
        ax3.view_init(elev=22, azim=-56)
        fig.savefig(output / "global_latent_pca_2d_3d.png", dpi=240)
        fig.savefig(output / "global_latent_pca_2d_3d.pdf")
        plt.close(fig)
    norm_summary = {}
    for name, array in zip(names, arrays):
        norms = np.linalg.norm(array, axis=1) / math.sqrt(array.shape[1])
        norm_summary[name] = {
            "mean_rms_norm": float(norms.mean()),
            "median_rms_norm": float(np.median(norms)),
            "std_rms_norm": float(norms.std()),
        }
    return {
        "samples_per_distribution": [len(x) for x in arrays],
        "latent_dimension": int(arrays[0].shape[1]),
        "pca_fit": "joint, equally sized native/cycle/expert arrays; no class reweighting needed",
        "explained_variance_ratio": explained.tolist(),
        "cumulative_explained_variance_first_2": float(explained[:2].sum()),
        "cumulative_explained_variance_first_3": float(explained.sum()),
        "rms_norm": norm_summary,
    }


def compute_jacobian(policy, matcher, z_star, condition, scale, steps):
    start = time.time()
    z = z_star.detach().clone().requires_grad_(True)
    raw = forward_flow(policy, matcher, z, condition, steps, recompute=True)
    y = action_vector(raw, policy, scale)[0]
    rows = []
    for index in range(y.numel()):
        grad = torch.autograd.grad(
            y[index], z, retain_graph=index + 1 < y.numel(), create_graph=False
        )[0]
        rows.append(grad.flatten().detach().cpu())
        if index == 0 or (index + 1) % 8 == 0 or index + 1 == y.numel():
            print(
                json.dumps(
                    {
                        "jacobian_rows": index + 1,
                        "total": y.numel(),
                        "elapsed_seconds": round(time.time() - start, 2),
                    }
                ),
                flush=True,
            )
    return torch.stack(rows).numpy(), time.time() - start


def tangent_spectrum(jacobian, radial):
    radial = radial / np.linalg.norm(radial)
    projector = np.eye(len(radial), dtype=np.float64) - np.outer(radial, radial)
    tangent_j = jacobian.astype(np.float64) @ projector
    _, singular_values, vh = np.linalg.svd(tangent_j, full_matrices=True)
    tolerance = singular_values[0] * 1e-4
    rank = int(np.sum(singular_values > tolerance))
    if rank < 2:
        raise RuntimeError(f"Tangent Jacobian numerical rank too low: {rank}")
    low_index = int(round(0.90 * (rank - 1)))
    high = vh[0]
    low = vh[low_index]
    # Reproject and Gram-Schmidt to make the construction auditable.
    high = high - radial * np.dot(high, radial)
    high /= np.linalg.norm(high)
    low = low - radial * np.dot(low, radial) - high * np.dot(low, high)
    low /= np.linalg.norm(low)
    return high, low, singular_values, rank, low_index, projector


@torch.no_grad()
def evaluate_plane(
    policy,
    matcher,
    z_star,
    condition,
    expert,
    scale,
    first,
    second,
    radius_rms,
    grid_size,
    batch_size,
    steps,
):
    latent_shape = tuple(z_star.shape[1:])
    dim = int(np.prod(latent_shape))
    axis_rms = np.linspace(-radius_rms, radius_rms, grid_size, dtype=np.float32)
    x_rms, y_rms = np.meshgrid(axis_rms, axis_rms)
    coefficients = np.stack([x_rms.ravel(), y_rms.ravel()], axis=1) * math.sqrt(dim)
    directions = torch.as_tensor(
        np.stack([first, second]), dtype=z_star.dtype, device=z_star.device
    )
    perturbations = torch.as_tensor(
        coefficients, dtype=z_star.dtype, device=z_star.device
    ) @ directions
    candidates = z_star.flatten(1) + perturbations
    center_raw = forward_flow(policy, matcher, z_star, condition, steps, recompute=False)
    center_vector = action_vector(center_raw, policy, scale)
    expert_vector = action_vector(expert, policy, scale)
    target_errors = []
    center_displacements = []
    for begin in range(0, len(candidates), batch_size):
        end = min(begin + batch_size, len(candidates))
        raw = forward_flow(
            policy,
            matcher,
            candidates[begin:end].reshape(-1, *latent_shape),
            condition.expand(end - begin, -1),
            steps,
            recompute=False,
        )
        vector = action_vector(raw, policy, scale)
        target_errors.extend((vector - expert_vector).norm(dim=1).cpu().tolist())
        center_displacements.extend((vector - center_vector).norm(dim=1).cpu().tolist())
        if begin == 0 or end == len(candidates) or (begin // batch_size) % 10 == 0:
            print(json.dumps({"grid_progress": end, "grid_total": len(candidates)}), flush=True)
    return {
        "axis_rms": axis_rms,
        "x_rms": x_rms,
        "y_rms": y_rms,
        "coefficients_l2": coefficients,
        "target_error": np.asarray(target_errors).reshape(grid_size, grid_size),
        "center_displacement": np.asarray(center_displacements).reshape(grid_size, grid_size),
        "absolute_rms_norm": candidates.norm(dim=1).cpu().numpy().reshape(grid_size, grid_size)
        / math.sqrt(dim),
        "center_target_error": float((center_vector - expert_vector).norm()),
    }


def positive_levels(values, count=8):
    positive = values[values > 1e-8]
    lo = max(float(np.quantile(positive, 0.02)), 1e-4)
    hi = float(np.quantile(positive, 0.98))
    return np.geomspace(lo, hi, count)


def plot_terrain(plane, output):
    x, y, error = plane["x_rms"], plane["y_rms"], plane["target_error"]
    levels = positive_levels(error, 80)
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    filled = ax.contourf(x, y, error, levels=levels, norm=LogNorm(), cmap="magma", extend="both")
    lines = ax.contour(x, y, error, levels=positive_levels(error, 7), colors="white", linewidths=0.65, alpha=0.72)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.2g")
    shell = ax.contour(x, y, plane["absolute_rms_norm"], levels=[1.0], colors="#41C7D9", linestyles="--", linewidths=1.5)
    if shell.allsegs[0]:
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D(
                    [0], [0], color="#41C7D9", linestyle="--", lw=1.5,
                    label=r"$\|z\|/\sqrt{d}=1$",
                )
            ],
            frameon=True,
            loc="upper left",
        )
    ax.scatter([0], [0], marker="*", s=145, c="#5DE2E7", edgecolor="black", linewidth=0.7, zorder=5)
    ax.annotate(r"$z^\star$", (0, 0), xytext=(7, 7), textcoords="offset points", color="white", weight="bold")
    ax.set_xlabel(r"high-sensitivity tangent $\beta/\sqrt{d}$")
    ax.set_ylabel(r"radial $\alpha/\sqrt{d}$  (outward $\uparrow$)")
    ax.set_title("Local behavior-region terrain")
    ax.set_aspect("equal")
    colorbar = fig.colorbar(filled, ax=ax, pad=0.02)
    colorbar.set_label(r"action error to $A^\star$")
    fig.savefig(output / "local_behavior_region_radial_high.png", dpi=260)
    fig.savefig(output / "local_behavior_region_radial_high.pdf")
    plt.close(fig)


def plot_anisotropy(plane, singular_high, singular_low, output):
    x, y = plane["x_rms"], plane["y_rms"]
    actual = plane["target_error"]
    dim = 144
    predicted = np.sqrt(
        (x * math.sqrt(dim) * singular_low) ** 2
        + (y * math.sqrt(dim) * singular_high) ** 2
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)
    common_hi = max(float(np.quantile(actual, 0.98)), float(np.quantile(predicted, 0.98)))
    common_lo = max(min(float(np.quantile(actual[actual > 1e-8], 0.02)), float(np.quantile(predicted[predicted > 1e-8], 0.02))), 1e-4)
    levels = np.geomspace(common_lo, common_hi, 80)
    contour_levels = np.geomspace(common_lo, common_hi, 7)
    for ax, values, title in zip(
        axes,
        [actual, predicted],
        ["Frozen Flow: action error to $A^\star$", r"Local metric: $\sqrt{\Delta z^\top G\Delta z}$"],
    ):
        filled = ax.contourf(x, y, values, levels=levels, norm=LogNorm(), cmap="viridis", extend="both")
        lines = ax.contour(x, y, values, levels=contour_levels, colors="white", linewidths=0.65, alpha=0.78)
        ax.clabel(lines, inline=True, fontsize=7, fmt="%.2g")
        ax.scatter([0], [0], marker="*", s=130, c="#FFDD55", edgecolor="black", linewidth=0.7, zorder=5)
        ax.set_xlabel(r"low-sensitivity tangent $\alpha/\sqrt{d}$")
        ax.set_ylabel(r"high-sensitivity tangent $\beta/\sqrt{d}$")
        ax.set_title(title)
        ax.set_aspect("equal")
    colorbar = fig.colorbar(filled, ax=axes, pad=0.02)
    colorbar.set_label(r"action error / local behavioral displacement")
    fig.suptitle("Tangential anisotropy: equal latent lengths, unequal behavior changes", fontsize=14)
    fig.savefig(output / "tangential_anisotropy_low_high.png", dpi=260)
    fig.savefig(output / "tangential_anisotropy_low_high.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    seed_all(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pca_report = make_pca_figures(args.global_npz, output, args.seed)
    print(json.dumps({"pca_complete": pca_report["explained_variance_ratio"]}), flush=True)

    data, manifest = load_cache(args.cache)
    if args.forward_steps != 200 or manifest["forward_steps"] != 200:
        raise ValueError("Verified comparison requires the native 200-step Flow")
    policy, matcher, _ = load_flow(args)
    if not all(not p.requires_grad and p.grad is None for p in policy.parameters()):
        raise AssertionError("Flow parameters are not frozen")
    val_idx = torch.where(data["split"] == 1)[0]
    direction_data = np.load(args.directional_npz)
    ordinal, selected = select_representative(direction_data)
    if len(val_idx) != len(direction_data["center_errors"]):
        raise ValueError("Directional arrays and cache validation split do not align")
    cache_index = int(val_idx[ordinal])
    selected.update(
        {
            "cache_index": cache_index,
            "sample_index": int(data["sample_index"][cache_index]),
            "condition_id": int(data["condition_id"][cache_index]),
            "episode": int(data["episode"][cache_index]),
        }
    )
    print(json.dumps({"representative_sample": selected}), flush=True)

    z_star = data["z_star"][cache_index : cache_index + 1].to(args.device)
    condition = data["condition"][cache_index : cache_index + 1].to(args.device)
    expert = data["expert"][cache_index : cache_index + 1].to(args.device)
    scale = action_scale(data).to(args.device)
    radial = z_star.flatten().detach().cpu().numpy().astype(np.float64)
    radial /= np.linalg.norm(radial)
    intermediate = output / "jacobian_intermediate.npz"
    if intermediate.exists():
        saved = np.load(intermediate)
        jacobian = saved["jacobian"]
        jacobian_seconds = float(saved["seconds"])
        print(json.dumps({"jacobian_reused": str(intermediate)}), flush=True)
    else:
        jacobian, jacobian_seconds = compute_jacobian(
            policy, matcher, z_star, condition, scale, args.forward_steps
        )
        np.savez_compressed(intermediate, jacobian=jacobian, seconds=jacobian_seconds)
    high, low, singular_values, rank, low_index, projector = tangent_spectrum(
        jacobian, radial
    )
    tangent_checks = {
        "radial_dot_high": float(np.dot(radial, high)),
        "radial_dot_low": float(np.dot(radial, low)),
        "high_dot_low": float(np.dot(high, low)),
        "high_norm": float(np.linalg.norm(high)),
        "low_norm": float(np.linalg.norm(low)),
    }
    print(json.dumps({"tangent_spectrum_rank": rank, "checks": tangent_checks}), flush=True)

    radial_high = evaluate_plane(
        policy,
        matcher,
        z_star,
        condition,
        expert,
        scale,
        high,
        radial,
        args.terrain_radius_rms,
        args.grid_size,
        args.grid_batch_size,
        args.forward_steps,
    )
    np.savez_compressed(
        output / "radial_high_intermediate.npz",
        axis_rms=radial_high["axis_rms"],
        x_rms=radial_high["x_rms"],
        y_rms=radial_high["y_rms"],
        target_error=radial_high["target_error"],
        center_displacement=radial_high["center_displacement"],
        absolute_rms_norm=radial_high["absolute_rms_norm"],
        center_target_error=radial_high["center_target_error"],
    )
    # evaluate_plane x=high, y=radial, matching the figure labels.
    if plt is not None:
        plot_terrain(radial_high, output)
    low_high = evaluate_plane(
        policy,
        matcher,
        z_star,
        condition,
        expert,
        scale,
        low,
        high,
        args.anisotropy_radius_rms,
        args.grid_size,
        args.grid_batch_size,
        args.forward_steps,
    )
    if plt is not None:
        plot_anisotropy(low_high, singular_values[0], singular_values[low_index], output)

    np.savez_compressed(
        output / "local_behavior_geometry_arrays.npz",
        jacobian=jacobian,
        tangent_projector=projector,
        tangent_singular_values=singular_values,
        radial_direction=radial,
        high_tangent_direction=high,
        low_tangent_direction=low,
        radial_high_axis_rms=radial_high["axis_rms"],
        radial_high_target_error=radial_high["target_error"],
        radial_high_center_displacement=radial_high["center_displacement"],
        radial_high_absolute_rms_norm=radial_high["absolute_rms_norm"],
        low_high_axis_rms=low_high["axis_rms"],
        low_high_target_error=low_high["target_error"],
        low_high_center_displacement=low_high["center_displacement"],
    )
    report = {
        "pca": pca_report,
        "representative_sample": selected,
        "frozen_flow": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "parameter_sha256": digest(policy),
            "forward_steps": args.forward_steps,
            "parameters_require_grad": False,
        },
        "jacobian": {
            "shape": list(jacobian.shape),
            "seconds": jacobian_seconds,
            "metric": "normalized executed action L2 / sqrt(n_action_steps)",
            "tangent_numerical_rank_relative_tol_1e-4": rank,
            "low_direction_rule": "90th percentile index of descending non-null singular spectrum",
            "low_direction_index": low_index,
            "sigma_high": float(singular_values[0]),
            "sigma_low": float(singular_values[low_index]),
            "condition_number_high_over_selected_low": float(
                singular_values[0] / singular_values[low_index]
            ),
            "lambda_ratio_high_over_low": float(
                (singular_values[0] / singular_values[low_index]) ** 2
            ),
            "direction_checks": tangent_checks,
        },
        "planes": {
            "radial_high": {
                "radius_rms": args.terrain_radius_rms,
                "grid_size": args.grid_size,
                "center_target_error": radial_high["center_target_error"],
                "outward_edge_median_error": float(np.median(radial_high["target_error"][-1])),
                "inward_edge_median_error": float(np.median(radial_high["target_error"][0])),
                "outward_over_inward_edge_median": float(
                    np.median(radial_high["target_error"][-1])
                    / np.median(radial_high["target_error"][0])
                ),
            },
            "low_high": {
                "radius_rms": args.anisotropy_radius_rms,
                "grid_size": args.grid_size,
                "center_target_error": low_high["center_target_error"],
            },
        },
    }
    write_json(report, output / "figure_report.json")
    print(json.dumps({"complete": str(output), "report": report["jacobian"]}), flush=True)


if __name__ == "__main__":
    main()
