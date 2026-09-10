"""Compare native N(0,I)->Flow->inverse-cycle latents with expert inversions."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA

from flow_latent_predictor_common import forward_flow, load_cache, load_flow, seed_all, sha256, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--reverse-steps", type=int, default=200)
    p.add_argument("--forward-steps", type=int, default=200)
    p.add_argument("--history", type=int, default=12)
    return p.parse_args()


def digest(module):
    import hashlib
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_cache(data, manifest, policy, args):
    if args.reverse_steps != 200 or args.forward_steps != 200:
        raise ValueError("This audit requires 200-step forward and reverse solvers")
    if manifest["reverse_steps"] != 200 or manifest["forward_steps"] != 200:
        raise ValueError("Cache must have 200-step inversion/reconstruction")
    if tuple(data["context"].shape[1:]) != (12, 521) or args.history != 12:
        raise ValueError("Expected causal context [12, 521]")
    if tuple(data["z_star"].shape[1:]) != (16, 9) or tuple(data["expert"].shape[1:]) != (16, 9):
        raise ValueError("Expected native latent/action shape [16, 9]")
    if policy.action_dim != 9 or policy.horizon != 16:
        raise ValueError("Checkpoint is not the native [16, 9] policy")
    if set(manifest["train_episodes"]) & set(manifest["val_episodes"]):
        raise AssertionError("Train/validation episode overlap")
    if not torch.all(data["observation_indices"] <= data["condition_id"][:, None]):
        raise AssertionError("Future observation leakage")
    if not torch.equal(data["observation_indices"][:, -1], data["condition_id"]):
        raise AssertionError("Context does not end at current timestep")
    if not torch.equal(data["condition"], data["context"][:, -8:].flatten(1)):
        raise AssertionError("Condition is not final eight context frames")
    if not all(not p.requires_grad and p.grad is None for p in policy.parameters()):
        raise AssertionError("Flow is not frozen before audit")


def zca_fit(x):
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=1)
    ev, u = np.linalg.eigh(cov)
    ev = np.maximum(ev, 1e-10)
    w = (u * (1.0 / np.sqrt(ev))) @ u.T
    return mean, cov, w


def whiten(x, mean, w):
    return (x - mean) @ w


def compact_stats(x, reference_whitened=None):
    n, d = x.shape
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=1)
    ev = np.maximum(np.linalg.eigvalsh(cov)[::-1], 0)
    prop = ev / max(ev.sum(), 1e-12)
    effective_rank = float(np.exp(-(prop[prop > 0] * np.log(prop[prop > 0])).sum()))
    norm2 = np.sum(x * x, axis=1)
    skew = stats.skew(x, axis=0, bias=False)
    kurt = stats.kurtosis(x, axis=0, fisher=True, bias=False)
    pca = PCA(n_components=10, svd_solver="full").fit(x)
    result = {
        "n": int(n), "dim": int(d), "mean_norm": float(np.linalg.norm(mean)),
        "global_mean": float(x.mean()), "global_std": float(x.std()),
        "cov_minus_I_fro": float(np.linalg.norm(cov - np.eye(d), ord="fro")),
        "cov_condition_number": float(ev[0] / max(ev[-1], 1e-12)),
        "effective_rank": effective_rank,
        "offdiag_cov_abs_mean": float(np.mean(np.abs(cov[~np.eye(d, dtype=bool)]))),
        "cov_eigenvalues_desc": ev.tolist(),
        "norm2": {"mean": float(norm2.mean()), "std": float(norm2.std(ddof=1)),
                   "p5": float(np.percentile(norm2, 5)), "p50": float(np.percentile(norm2, 50)),
                   "p95": float(np.percentile(norm2, 95)), "chi2_mean": float(d),
                   "chi2_std": float(math.sqrt(2 * d))},
        "gaussianity": {"skew_mean": float(skew.mean()), "skew_max_abs": float(np.max(np.abs(skew))),
                        "kurtosis_excess_mean": float(kurt.mean()), "kurtosis_excess_max_abs": float(np.max(np.abs(kurt)))},
        "pca_explained_variance_ratio_top10": pca.explained_variance_ratio_.tolist(),
        "pca_top1_over_uniform": float(pca.explained_variance_ratio_[0] / (1 / d)),
    }
    if reference_whitened is not None:
        rw = reference_whitened
        rw_norm2 = np.sum(rw * rw, axis=1)
        rw_cov = np.cov(rw, rowvar=False, ddof=1)
        rw_skew = stats.skew(rw, axis=0, bias=False)
        rw_kurt = stats.kurtosis(rw, axis=0, fisher=True, bias=False)
        result["common_expert_whitened"] = {
            "mean_norm": float(np.linalg.norm(rw.mean(0))),
            "per_dim_mean_max_abs": float(np.max(np.abs(rw.mean(0)))),
            "per_dim_std_min": float(rw.std(0, ddof=1).min()),
            "per_dim_std_max": float(rw.std(0, ddof=1).max()),
            "cov_minus_I_fro": float(np.linalg.norm(rw_cov - np.eye(d), ord="fro")),
            "norm2": {"mean": float(rw_norm2.mean()), "std": float(rw_norm2.std(ddof=1)),
                       "p5": float(np.percentile(rw_norm2, 5)), "p50": float(np.percentile(rw_norm2, 50)),
                       "p95": float(np.percentile(rw_norm2, 95)), "chi2_mean": float(d),
                       "chi2_std": float(math.sqrt(2 * d))},
            "gaussianity": {"skew_mean": float(rw_skew.mean()), "skew_max_abs": float(np.max(np.abs(rw_skew))),
                            "kurtosis_excess_mean": float(rw_kurt.mean()), "kurtosis_excess_max_abs": float(np.max(np.abs(rw_kurt)))},
        }
    return result


def audit_group(name, x, train_mask, expert_mean, expert_w, output):
    train = x[train_mask]
    common = whiten(train, expert_mean, expert_w)
    own_mean, own_cov, own_w = zca_fit(train)
    own_white = whiten(train, own_mean, own_w)
    result = compact_stats(train, common)
    result["self_whitened"] = compact_stats(own_white)
    np.savez_compressed(output / f"{name}_audit_arrays.npz", raw_train=train, common_expert_whitened=common,
                        self_whitened=own_white, mean=own_mean, covariance=own_cov, whitening_matrix=own_w)
    return result


def plot_audit(groups, output):
    labels = list(groups)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for label in labels:
        d = groups[label]
        axes[0].bar(label, d["norm2"]["mean"]); axes[1].bar(label, d["common_expert_whitened"]["norm2"]["mean"])
        axes[2].bar(label, d["gaussianity"]["kurtosis_excess_mean"])
    axes[0].axhline(144, color="black", linestyle="--"); axes[1].axhline(144, color="black", linestyle="--")
    axes[0].set_title("raw ||z||^2 mean"); axes[1].set_title("common expert-whitened ||z||^2 mean"); axes[2].set_title("raw mean excess kurtosis")
    for ax in axes: ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(output / "distribution_comparison.png", dpi=180); plt.close(fig)


def main():
    args = parse_args(); seed_all(args.seed); args.device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache); policy, matcher, _ = load_flow(args); check_cache(data, manifest, policy, args)
    before = digest(policy); n = len(data["z_star"]); native_parts=[]; cycle_parts=[]; generation_rmse=[]; cycle_rmse=[]; cycle_latent_rmse=[]
    print(json.dumps({"status":"starting","samples":n,"batch_size":args.batch_size,"device":args.device}), flush=True)
    with torch.no_grad():
        for begin in range(0, n, args.batch_size):
            end = min(n, begin + args.batch_size); b = end - begin
            condition = data["condition"][begin:end].to(args.device)
            z_native = torch.randn((b, 16, 9), device=args.device)
            generated = forward_flow(policy, matcher, z_native, condition, args.forward_steps, recompute=False)
            z_cycle = matcher.reverse_sample(policy.model, start=generated, num_steps=args.reverse_steps, global_cond=condition)
            reconstructed = forward_flow(policy, matcher, z_cycle, condition, args.forward_steps, recompute=False)
            native_parts.append(z_native.cpu()); cycle_parts.append(z_cycle.cpu())
            generation_rmse.append(generated.flatten(1).square().mean(1).sqrt().cpu())
            cycle_rmse.append((reconstructed - generated).flatten(1).square().mean(1).sqrt().cpu())
            cycle_latent_rmse.append((z_cycle - z_native).flatten(1).square().mean(1).sqrt().cpu())
            if end == n or (begin // args.batch_size) % 20 == 0:
                print(json.dumps({"progress": end, "total": n, "generated_action_rms_mean": float(torch.cat(generation_rmse).mean()),
                                  "cycle_action_rmse_mean": float(torch.cat(cycle_rmse).mean()),
                                  "cycle_latent_rmse_mean": float(torch.cat(cycle_latent_rmse).mean())}), flush=True)
    native = torch.cat(native_parts).numpy().astype(np.float64); cycle = torch.cat(cycle_parts).numpy().astype(np.float64); expert = data["z_star"].numpy().reshape(n, -1).astype(np.float64)
    native = native.reshape(n, -1); cycle = cycle.reshape(n, -1)
    train_mask = (data["split"].numpy() == 0); val_mask = ~train_mask
    expert_train = expert[train_mask]; expert_mean, expert_cov, expert_w = zca_fit(expert_train)
    groups = {"native": audit_group("native", native, train_mask, expert_mean, expert_w, output),
              "cycle": audit_group("cycle", cycle, train_mask, expert_mean, expert_w, output),
              "expert": audit_group("expert", expert, train_mask, expert_mean, expert_w, output)}
    heldout = {}
    for name, x in (("native", native), ("cycle", cycle), ("expert", expert)):
        xv = x[val_mask]; common = whiten(xv, expert_mean, expert_w); heldout[name] = compact_stats(xv, common)
    np.savez_compressed(output / "native_cycle_expert_all.npz", native=native, cycle=cycle, expert=expert,
                        train_mask=train_mask, val_mask=val_mask, expert_mean=expert_mean, expert_cov=expert_cov, expert_whitening=expert_w)
    cycle_report = {"generated_action_rms_mean": float(torch.cat(generation_rmse).mean()), "generated_action_rms_p95": float(torch.quantile(torch.cat(generation_rmse), .95)),
                    "cycle_reconstruction_action_rmse_mean": float(torch.cat(cycle_rmse).mean()), "cycle_reconstruction_action_rmse_p95": float(torch.quantile(torch.cat(cycle_rmse), .95)),
                    "native_cycle_latent_rmse_mean": float(torch.cat(cycle_latent_rmse).mean()), "native_cycle_latent_rmse_p95": float(torch.quantile(torch.cat(cycle_latent_rmse), .95))}
    after = digest(policy)
    report = {"status":"ok","n_total":n,"n_train":int(train_mask.sum()),"n_val":int(val_mask.sum()),"shape":[16,9],"flatten_dim":144,
              "seed":args.seed,"forward_steps":args.forward_steps,"reverse_steps":args.reverse_steps,"flow_checkpoint_sha256":sha256(args.checkpoint),
              "cache_manifest_sha256":sha256(Path(args.cache)/"manifest.json"),"cycle_diagnostics":cycle_report,"train_audit":groups,"val_audit_using_train_expert_whitener":heldout,
              "flow_parameter_digest_before":before,"flow_parameter_digest_after":after,"flow_unchanged":before==after,"all_flow_parameter_grads_none":all(p.grad is None for p in policy.parameters())}
    write_json(report, output / "audit.json"); write_json({"args":vars(args),"report_file":"audit.json"}, output / "config.json"); plot_audit(groups, output)
    print(json.dumps({"cycle_diagnostics":cycle_report,"native_train_norm2":groups["native"]["norm2"],"cycle_train_norm2":groups["cycle"]["norm2"],"expert_train_norm2":groups["expert"]["norm2"],"native_common_whiten":groups["native"]["common_expert_whitened"],"cycle_common_whiten":groups["cycle"]["common_expert_whitened"],"expert_common_whiten":groups["expert"]["common_expert_whitened"],"flow_unchanged":before==after},indent=2),flush=True)


if __name__ == "__main__":
    main()
