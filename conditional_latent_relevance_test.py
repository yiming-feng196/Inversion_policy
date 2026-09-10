"""Measure how much the frozen Flow output depends on source latent under fixed c."""
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

from flow_latent_predictor_common import forward_flow, load_cache, load_flow, seed_all, sha256, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--batch-size", type=int, default=4, help="Validation queries per Flow batch group")
    p.add_argument("--random-samples", type=int, default=100)
    p.add_argument("--forward-steps", type=int, default=200)
    p.add_argument("--reverse-steps", type=int, default=200)
    p.add_argument("--max-val-samples", type=int, default=0, help="0 means all validation samples")
    return p.parse_args()


def digest(module):
    import hashlib
    h = hashlib.sha256()
    for name, parameter in module.named_parameters():
        h.update(name.encode()); h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_settings(data, manifest, policy, args):
    if args.forward_steps != 200 or args.reverse_steps != 200:
        raise ValueError("This test requires 200-step Flow evaluation")
    if manifest["forward_steps"] != 200 or manifest["reverse_steps"] != 200:
        raise ValueError("Cache solver settings are not 200/200")
    if tuple(data["context"].shape[1:]) != (12, 521):
        raise ValueError("Expected verified 12-frame context")
    if tuple(data["z_star"].shape[1:]) != (16, 9) or tuple(data["expert"].shape[1:]) != (16, 9):
        raise ValueError("Expected native [16, 9] latent/action shape")
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
    return data["expert_raw"][data["split"] == 0].flatten(0, 1).std(0, unbiased=False).clamp_min(1e-6)


def normalized_action(action, policy, scale):
    return policy.normalizer["action"].unnormalize(action) / scale.to(action.device)


def distances(pred, target, policy, scale):
    d = normalized_action(pred, policy, scale) - normalized_action(target, policy, scale)
    full = d.square().sum(-1).mean(-1).sqrt()
    start = policy.n_obs_steps - 1
    executed = d[:, start:start + policy.n_action_steps]
    executed_distance = executed.square().sum(-1).mean(-1).sqrt()
    return full, executed_distance, executed.flatten(1)


@torch.no_grad()
def evaluate(args, data, policy, matcher, val_idx, scale, output):
    n = len(val_idx); m = args.random_samples
    generator = torch.Generator().manual_seed(args.seed)
    all_rows = []; curve = {k: [] for k in (1, 2, 4, 8, 16, 32, 64, 100) if k <= m}
    inversion_full=[]; inversion_exec=[]; random_full_mean=[]; random_exec_mean=[]; random_diversity=[]; ratios=[]
    random_best8=[]; random_best100=[]
    for begin in range(0, n, args.batch_size):
        ids = val_idx[begin:begin + args.batch_size]; b = len(ids)
        condition = data["condition"][ids].to(args.device); expert = data["expert"][ids].to(args.device); z_star = data["z_star"][ids].to(args.device)
        inv_action = forward_flow(policy, matcher, z_star, condition, args.forward_steps, recompute=False)
        inv_full, inv_exec, _ = distances(inv_action, expert, policy, scale)
        z_rand = torch.randn((b, m, 16, 9), generator=generator).to(args.device)
        random_action = forward_flow(policy, matcher, z_rand.reshape(b * m, 16, 9), condition[:, None].expand(b, m, -1).reshape(b * m, -1), args.forward_steps, recompute=False).reshape(b, m, 16, 9)
        target = expert[:, None].expand_as(random_action)
        rand_full, rand_exec, rand_vectors = distances(random_action.reshape(b * m, 16, 9), target.reshape(b * m, 16, 9), policy, scale)
        rand_full = rand_full.reshape(b, m); rand_exec = rand_exec.reshape(b, m); rand_vectors = rand_vectors.reshape(b, m, -1)
        pairwise = torch.cdist(rand_vectors, rand_vectors) / math.sqrt(policy.n_action_steps)
        diversity = (pairwise.sum(-1) / max(1, m - 1)).mean(-1)
        best = rand_exec.min(-1).values
        ratio = inv_exec / rand_exec.median(-1).values.clamp_min(1e-12)
        for k in curve: curve[k].extend(rand_exec[:, :k].min(-1).values.cpu().tolist())
        inversion_full.extend(inv_full.cpu().tolist()); inversion_exec.extend(inv_exec.cpu().tolist())
        random_full_mean.extend(rand_full.mean(-1).cpu().tolist()); random_exec_mean.extend(rand_exec.mean(-1).cpu().tolist())
        random_diversity.extend(diversity.cpu().tolist()); ratios.extend(ratio.cpu().tolist()); random_best8.extend(rand_exec[:, :min(8, m)].min(-1).values.cpu().tolist()); random_best100.extend(best.cpu().tolist())
        for i, q in enumerate(ids.tolist()):
            all_rows.append({"sample_index": q, "episode_id": int(data["episode"][q]), "inversion_full_error": float(inv_full[i]), "inversion_executed_error": float(inv_exec[i]),
                             "random_full_mean": float(rand_full[i].mean()), "random_executed_mean": float(rand_exec[i].mean()), "random_best8_executed": float(rand_exec[i, :min(8, m)].min()),
                             "random_best100_executed": float(best[i]), "random_action_diversity": float(diversity[i]), "inversion_over_random_median": float(ratio[i])})
        if begin == 0 or (begin // args.batch_size) % 25 == 0 or begin + b == n:
            print(json.dumps({"progress": begin + b, "total": n, "inversion_exec_mean": float(np.mean(inversion_exec)), "random_exec_mean": float(np.mean(random_exec_mean)), "random_best100_mean": float(np.mean(random_best100))}), flush=True)
    rows_array = {k: np.asarray(v, dtype=np.float64) for k, v in {"inversion_full": inversion_full, "inversion_executed": inversion_exec, "random_full_mean": random_full_mean, "random_executed_mean": random_exec_mean, "random_diversity": random_diversity, "ratio": ratios, "random_best8": random_best8, "random_best100": random_best100}.items()}
    curve_summary = {str(k): {"mean": float(np.mean(v)), "median": float(np.median(v)), "p95": float(np.quantile(v, .95))} for k, v in curve.items()}
    report = {"n_validation": n, "random_samples_per_condition": m, "metrics": {"inversion": {"full_mean": float(rows_array["inversion_full"].mean()), "full_median": float(np.median(rows_array["inversion_full"])), "executed_mean": float(rows_array["inversion_executed"].mean()), "executed_median": float(np.median(rows_array["inversion_executed"]))},
        "random_single": {"full_mean": float(rows_array["random_full_mean"].mean()), "executed_mean": float(rows_array["random_executed_mean"].mean())},
        "random_best8": {"executed_mean": float(rows_array["random_best8"].mean()), "executed_median": float(np.median(rows_array["random_best8"]))},
        "random_best100": {"executed_mean": float(rows_array["random_best100"].mean()), "executed_median": float(np.median(rows_array["random_best100"]))},
        "random_same_condition_diversity": {"mean": float(rows_array["random_diversity"].mean()), "median": float(np.median(rows_array["random_diversity"])), "p95": float(np.quantile(rows_array["random_diversity"], .95))},
        "inversion_over_random_median_ratio": {"mean": float(rows_array["ratio"].mean()), "median": float(np.median(rows_array["ratio"])), "p95": float(np.quantile(rows_array["ratio"], .95))}},
        "best_of_k_curve": curve_summary,
        "interpretation": {"latent_relevance": "strong if inversion executed error is much smaller than random single and random diversity is nonzero", "localization_value": "strong if random best@100 remains above inversion", "condition_dependence_control": "not included in this fixed-condition test; use shuffle only for conditional predictors"}}
    np.savez_compressed(output / "relevance_arrays.npz", **rows_array)
    with (output / "best_of_k_curve.csv").open("w") as f:
        f.write("k,mean,median,p95\n")
        for k, s in curve_summary.items(): f.write(f"{k},{s['mean']},{s['median']},{s['p95']}\n")
    write_rows(output / "sample_level_results.csv", all_rows)
    return report


def write_rows(path, rows):
    import csv
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args(); seed_all(args.seed); args.device = str(torch.device(args.device if torch.cuda.is_available() else "cpu")); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache); policy, matcher, _ = load_flow(args); check_settings(data, manifest, policy, args); before = digest(policy)
    val_idx = torch.where(data["split"] == 1)[0]
    if args.max_val_samples: val_idx = val_idx[:args.max_val_samples]
    report = evaluate(args, data, policy, matcher, val_idx, action_scale(data), output)
    after = digest(policy); report.update({"status": "ok", "seed": args.seed, "forward_steps": args.forward_steps, "reverse_steps": args.reverse_steps, "shape": [16, 9], "flow_checkpoint_sha256": sha256(args.checkpoint), "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"), "flow_parameter_digest_before": before, "flow_parameter_digest_after": after, "flow_unchanged": before == after, "all_flow_parameter_grads_none": all(p.grad is None for p in policy.parameters())})
    write_json(report, output / "relevance_report.json"); write_json({"args": vars(args), "report_file": "relevance_report.json"}, output / "config.json")
    metrics = report["metrics"]; labels = ["inversion", "random single", "random best@8", "random best@100"]; values = [metrics["inversion"]["executed_mean"], metrics["random_single"]["executed_mean"], metrics["random_best8"]["executed_mean"], metrics["random_best100"]["executed_mean"]]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(labels, values); ax.set_ylabel("normalized executed action distance"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(output / "fixed_condition_error_comparison.png", dpi=180); plt.close(fig)
    curve = report["best_of_k_curve"]; fig, ax = plt.subplots(figsize=(7, 5)); ks = [int(k) for k in curve]; ax.plot(ks, [curve[str(k)]["mean"] for k in ks], marker="o"); ax.set(xscale="log", xlabel="random samples K", ylabel="best-of-K executed action distance"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "best_of_k_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.hist(report["metrics"]["random_same_condition_diversity"].get("values", []), bins=30) if False else None
    print(json.dumps({"metrics": metrics, "best_of_k_curve": curve, "flow_unchanged": report["flow_unchanged"]}, indent=2), flush=True)


if __name__ == "__main__": main()
