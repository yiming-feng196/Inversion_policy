"""Train and evaluate a train-only Behavior Region Localization (BRL) model.

The script deliberately separates two operations:

* the locator is trained only on train-episode observation features and train
  inversion latents;
* validation behavior errors are used only for offline diagnostics.

Candidate labels are obtained by decoding a train-bank latent under the
current query condition.  This makes the supervision behavioral compatibility
rather than episode identity or Euclidean latent distance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flow_latent_predictor_common import (
    atomic_save,
    forward_flow,
    load_cache,
    load_flow,
    seed_all,
    sha256,
    write_json,
)


def digest(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, parameter in module.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--flow-batch-size", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=128)
    parser.add_argument("--observation-neighbors", type=int, default=64)
    parser.add_argument("--proprio-neighbors", type=int, default=32)
    parser.add_argument("--random-candidates", type=int, default=32)
    parser.add_argument("--teacher-label-steps", type=int, default=50)
    parser.add_argument("--teacher-fallback-steps", type=int, default=100)
    parser.add_argument("--reference-steps", type=int, default=200)
    parser.add_argument("--ranking-pairs", type=int, default=5000)
    parser.add_argument("--teacher-min-spearman", type=float, default=0.90)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--locator-batch-size", type=int, default=32)
    parser.add_argument("--locator-hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--tau-teacher", type=float, default=0.02)
    parser.add_argument("--tau-locator", type=float, default=0.07)
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--region-sigma", type=float, default=0.25)
    parser.add_argument("--region-samples", type=int, default=8)
    parser.add_argument("--region-query-limit", type=int, default=512)
    return parser.parse_args()


def ensure_tensor(data: dict, key: str, dtype=torch.float32) -> torch.Tensor:
    if key not in data:
        raise KeyError(f"cache is missing required field {key!r}")
    return data[key].to(dtype=dtype)


def check_settings(
    data: dict,
    manifest: dict,
    policy: nn.Module,
    args: argparse.Namespace,
) -> None:
    if args.reference_steps != 200:
        raise ValueError("formal BRL validation must use --reference-steps 200")
    if tuple(data["z_star"].shape[1:]) != (16, 9):
        raise ValueError(f"expected latent shape [16,9], got {tuple(data['z_star'].shape[1:])}")
    if tuple(data["expert"].shape[1:]) != (16, 9):
        raise ValueError(f"expected action shape [16,9], got {tuple(data['expert'].shape[1:])}")
    if tuple(data["context"].shape[1:]) != (12, 521):
        raise ValueError(f"expected 12-frame encoded context [12,521], got {tuple(data['context'].shape[1:])}")
    if policy.horizon != 16 or policy.action_dim != 9:
        raise ValueError("checkpoint must have horizon=16 and action_dim=9")
    if manifest.get("forward_steps") != 200 or manifest.get("reverse_steps") != 200:
        raise ValueError("the inversion cache must be generated with 200-step Flow")
    if set(manifest["train_episodes"]) & set(manifest["val_episodes"]):
        raise AssertionError("train and validation episodes overlap")
    if not torch.all(data["observation_indices"] <= data["condition_id"][:, None]):
        raise AssertionError("future observation leakage detected")
    if not torch.equal(data["observation_indices"][:, -1], data["condition_id"]):
        raise AssertionError("observation history does not end at current timestep")
    if not torch.equal(data["condition"], data["context"][:, -8:].flatten(1)):
        raise AssertionError("Flow condition is not the final eight observation frames")
    if any(parameter.requires_grad or parameter.grad is not None for parameter in policy.parameters()):
        raise AssertionError("Flow must be frozen and have no parameter gradients")


def train_mask(data: dict) -> torch.Tensor:
    return data["split"] == 0


def validation_mask(data: dict) -> torch.Tensor:
    return data["split"] == 1


def build_bank(data: dict, output: Path) -> dict:
    """Create a bank containing train episodes only."""
    ids = torch.where(train_mask(data))[0]
    context = data["context"][ids].float().contiguous()
    bank = {
        "sample_index": data["sample_index"][ids].long().contiguous(),
        "episode_id": data["episode"][ids].long().contiguous(),
        "timestep": data["condition_id"][ids].long().contiguous(),
        "condition_feature": context.flatten(1),
        "condition": data["condition"][ids].float().contiguous(),
        "z_star": data["z_star"][ids].float().contiguous(),
        "latent_norm": data["z_norm"][ids].float().contiguous(),
        "expert_action": data["expert"][ids].float().contiguous(),
        "context": context,
    }
    if "proprio" in data:
        bank["proprio"] = data["proprio"][ids].float().contiguous()
        bank["proprio_available"] = True
    else:
        bank["proprio"] = torch.zeros((len(ids), 0), dtype=torch.float32)
        bank["proprio_available"] = False
    for tau_key in ("x_tau_025", "x_tau_050", "x_tau_075"):
        if tau_key in data:
            bank[tau_key] = data[tau_key][ids].float().contiguous()
    atomic_save(bank, output / "bank.pt")
    return bank


def _normalise_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _top_indices(query: torch.Tensor, bank: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or bank.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long)
    score = _normalise_rows(query[None]) @ _normalise_rows(bank).T
    return torch.topk(score[0], k=min(count, bank.shape[0]), largest=True).indices.cpu()


def build_candidate_pool(
    data: dict,
    bank: dict,
    query_ids: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return train-bank candidate indices and source labels.

    Source labels are 0=self, 1=observation-near, 2=proprio-near and
    3=random.  Phase is never used as an input or a target.
    """
    rng = np.random.default_rng(args.seed + 17)
    bank_features = bank["condition_feature"]
    query_features = data["context"].flatten(1)
    bank_proprio = bank["proprio"]
    query_proprio = data.get("proprio")
    use_proprio = bool(bank["proprio_available"]) and query_proprio is not None and query_proprio.shape[-1] > 0
    bank_sample_to_pos = {int(x): i for i, x in enumerate(bank["sample_index"].tolist())}
    pool_size = min(args.candidate_pool, len(bank_features))
    n_obs = min(args.observation_neighbors, pool_size)
    n_prop = min(args.proprio_neighbors, max(0, pool_size - n_obs))
    n_rand = min(args.random_candidates, max(0, pool_size - n_obs - n_prop))
    while n_obs + n_prop + n_rand < pool_size:
        n_rand += 1

    candidates, sources = [], []
    norm_bank_feature = _normalise_rows(bank_features)
    norm_bank_prop = _normalise_rows(bank_proprio) if use_proprio else None
    for q in query_ids.tolist():
        q_feature = query_features[q].flatten()
        obs_order = (norm_bank_feature @ _normalise_rows(q_feature[None])[0]).argsort(descending=True).tolist()
        prop_order = []
        if use_proprio:
            prop_order = (norm_bank_prop @ _normalise_rows(query_proprio[q][None])[0]).argsort(descending=True).tolist()
        self_pos = bank_sample_to_pos.get(int(q))
        selected: list[int] = []
        source: list[int] = []

        def add(indices: Iterable[int], source_id: int, limit: int) -> None:
            for candidate in indices:
                candidate = int(candidate)
                if len(selected) >= pool_size or sum(s == source_id for s in source) >= limit:
                    break
                if candidate not in selected:
                    selected.append(candidate)
                    source.append(source_id)

        if self_pos is not None:
            selected.append(self_pos)
            source.append(0)
        add(obs_order, 1, n_obs)
        if use_proprio:
            add(prop_order, 2, n_prop)
        random_order = rng.permutation(len(bank_features)).tolist()
        add(random_order, 3, n_rand)
        add(range(len(bank_features)), 3, pool_size)
        candidates.append(selected[:pool_size])
        sources.append(source[:pool_size])
    return torch.tensor(candidates, dtype=torch.long), torch.tensor(sources, dtype=torch.long)


def action_scale(data: dict) -> torch.Tensor:
    raw = data["expert_raw"][train_mask(data)].flatten(0, 1).float()
    return raw.std(0, unbiased=False).clamp_min(1e-6)


def executed_action_error(
    action: torch.Tensor,
    target: torch.Tensor,
    policy: nn.Module,
    scale: torch.Tensor,
) -> torch.Tensor:
    action = policy.normalizer["action"].unnormalize(action)
    target = policy.normalizer["action"].unnormalize(target)
    delta = (action - target) / scale.to(action.device)
    start = int(policy.n_obs_steps) - 1
    stop = start + int(policy.n_action_steps)
    if start != 7 or stop != 15:
        raise ValueError(f"unexpected executed slice [{start}:{stop}], expected [7:15]")
    delta = delta[:, start:stop]
    return delta.square().sum(-1).mean(-1).sqrt()


@torch.no_grad()
def decode_pair_errors(
    policy: nn.Module,
    matcher,
    data: dict,
    bank: dict,
    query_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    scale: torch.Tensor,
    steps: int,
    device: str,
    query_batch_size: int,
    flow_batch_size: int,
) -> np.ndarray:
    """Decode candidates under each current query condition."""
    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [queries,candidates]")
    values = []
    for begin in range(0, len(query_ids), query_batch_size):
        qids = query_ids[begin:begin + query_batch_size]
        cids = candidate_ids[begin:begin + query_batch_size]
        batch, count = cids.shape
        condition = data["condition"][qids].float().to(device)
        target = data["expert"][qids].float().to(device)
        z = bank["z_star"][cids].float()
        flat_z = z.reshape(batch * count, *z.shape[2:])
        flat_condition = condition[:, None].expand(batch, count, -1).reshape(batch * count, -1)
        flat_target = target[:, None].expand(batch, count, *target.shape[1:]).reshape(batch * count, *target.shape[1:])
        error_chunks = []
        for off in range(0, len(flat_z), flow_batch_size):
            prediction = forward_flow(
                policy,
                matcher,
                flat_z[off:off + flow_batch_size].to(device),
                flat_condition[off:off + flow_batch_size],
                steps,
                recompute=False,
            )
            error_chunks.append(executed_action_error(prediction, flat_target[off:off + flow_batch_size], policy, scale).cpu())
        values.append(torch.cat(error_chunks).reshape(batch, count))
        print(json.dumps({"decoded_queries": min(begin + batch, len(query_ids)), "total_queries": len(query_ids), "steps": steps}), flush=True)
    return torch.cat(values).numpy()


@torch.no_grad()
def decode_direct_errors(
    policy: nn.Module,
    matcher,
    data: dict,
    query_ids: torch.Tensor,
    z: torch.Tensor,
    scale: torch.Tensor,
    steps: int,
    device: str,
    query_batch_size: int,
    flow_batch_size: int,
) -> np.ndarray:
    """Decode one or more latent samples per query."""
    if z.ndim != 4:
        raise ValueError("z must have shape [queries,samples,16,9]")
    values = []
    for begin in range(0, len(query_ids), query_batch_size):
        qids = query_ids[begin:begin + query_batch_size]
        batch = len(qids)
        count = z.shape[1]
        condition = data["condition"][qids].float().to(device)
        target = data["expert"][qids].float().to(device)
        flat_z = z[begin:begin + batch].reshape(batch * count, *z.shape[2:])
        flat_condition = condition[:, None].expand(batch, count, -1).reshape(batch * count, -1)
        flat_target = target[:, None].expand(batch, count, *target.shape[1:]).reshape(batch * count, *target.shape[1:])
        error_chunks = []
        for off in range(0, len(flat_z), flow_batch_size):
            prediction = forward_flow(
                policy,
                matcher,
                flat_z[off:off + flow_batch_size].to(device),
                flat_condition[off:off + flow_batch_size],
                steps,
                recompute=False,
            )
            error_chunks.append(executed_action_error(prediction, flat_target[off:off + flow_batch_size], policy, scale).cpu())
        values.append(torch.cat(error_chunks).reshape(batch, count))
    return torch.cat(values).numpy()


def rank_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman


def teacher_ranking_report(
    label: np.ndarray,
    reference: np.ndarray,
    min_pairs: int,
) -> dict:
    count = label.shape[1]
    n_queries = min(label.shape[0], max(1, math.ceil(min_pairs / count)))
    a = label[:n_queries]
    b = reference[:n_queries]
    pearson, spearman = rank_correlation(a, b)
    report = {
        "n_queries": int(n_queries),
        "n_pairs": int(a.size),
        "pearson": pearson,
        "spearman": spearman,
    }
    report["top1_agreement"] = float(
        np.mean(np.argsort(a, axis=1)[:, 0] == np.argsort(b, axis=1)[:, 0])
    )
    for k in (1, 4, 8):
        kk = min(k, count)
        a_top = np.argsort(a, axis=1)[:, :kk]
        b_top = np.argsort(b, axis=1)[:, :kk]
        overlaps = [len(set(x.tolist()) & set(y.tolist())) / kk for x, y in zip(a_top, b_top)]
        report[f"top{k}_overlap"] = float(np.mean(overlaps))
    return report


def estimate_region_epsilon(
    args: argparse.Namespace,
    data: dict,
    policy: nn.Module,
    matcher,
    query_ids: torch.Tensor,
    scale: torch.Tensor,
    device: str,
) -> tuple[float, dict]:
    limit = len(query_ids) if args.region_query_limit <= 0 else min(args.region_query_limit, len(query_ids))
    query_ids = query_ids[:limit]
    generator = torch.Generator().manual_seed(args.seed + 31)
    z_star = data["z_star"][query_ids].float()
    noise = torch.randn(
        (len(query_ids), args.region_samples, *z_star.shape[1:]),
        generator=generator,
        dtype=z_star.dtype,
    )
    z_local = z_star[:, None] + args.region_sigma * noise
    errors = decode_direct_errors(
        policy, matcher, data, query_ids, z_local, scale,
        args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    epsilon = float(np.quantile(errors, 0.95))
    return epsilon, {
        "sigma": args.region_sigma,
        "samples_per_query": args.region_samples,
        "queries": len(query_ids),
        "error_mean": float(errors.mean()),
        "error_median": float(np.median(errors)),
        "error_p95": epsilon,
    }


class RegionLocator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-6))
        self.query_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.key_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def normalise(self, features: torch.Tensor) -> torch.Tensor:
        return (features.float() - self.feature_mean) / self.feature_std

    def encode_query(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_net(self.normalise(features)), dim=-1)

    def encode_key(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_net(self.normalise(features)), dim=-1)


def train_locator(
    args: argparse.Namespace,
    data: dict,
    bank: dict,
    query_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    teacher_errors: np.ndarray,
    device: str,
    output: Path,
) -> tuple[RegionLocator, list[dict]]:
    features = bank["condition_feature"].float()
    feature_mean = features.mean(0)
    feature_std = features.std(0, unbiased=False).clamp_min(1e-5)
    model = RegionLocator(
        features.shape[-1], args.locator_hidden_dim, args.embedding_dim,
        feature_mean, feature_std,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    query_features = data["context"][query_ids].flatten(1).float()
    candidate_ids_t = candidate_ids.long()
    labels = torch.as_tensor(teacher_errors, dtype=torch.float32)
    history = []
    generator = torch.Generator().manual_seed(args.seed + 47)
    for epoch in range(args.train_epochs):
        order = torch.randperm(len(query_ids), generator=generator)
        epoch_losses = []
        for begin in range(0, len(order), args.locator_batch_size):
            rows = order[begin:begin + args.locator_batch_size]
            q = model.encode_query(query_features[rows].to(device))
            keys = bank["condition_feature"][candidate_ids_t[rows]].to(device)
            k = model.encode_key(keys.reshape(-1, keys.shape[-1])).reshape(len(rows), -1, args.embedding_dim)
            logits = (q[:, None] * k).sum(-1) / args.tau_locator
            target = F.softmax(-labels[rows].to(device) / args.tau_teacher, dim=-1)
            loss = -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(epoch_losses))}
        history.append(row)
        print(json.dumps(row), flush=True)
    atomic_save({
        "state_dict": model.state_dict(),
        "input_dim": int(features.shape[-1]),
        "hidden_dim": args.locator_hidden_dim,
        "embedding_dim": args.embedding_dim,
        "tau_locator": args.tau_locator,
        "tau_teacher": args.tau_teacher,
        "train_queries": query_ids,
    }, output / "locator.pt")
    write_rows(output / "locator_training.csv", history)
    return model, history


def _method_summary(errors: np.ndarray, top_indices: np.ndarray, epsilon: float) -> dict:
    top_indices = np.asarray(top_indices)
    selected = np.take_along_axis(errors, top_indices, axis=1)
    result = {
        "top1_mean": float(selected[:, 0].mean()),
        "top1_median": float(np.median(selected[:, 0])),
        "top1_p90": float(np.quantile(selected[:, 0], 0.90)),
        "region_hit@1": float(np.mean(selected[:, :1] <= epsilon)),
    }
    for k in (4, 8):
        kk = min(k, selected.shape[1])
        result[f"best@{k}_mean"] = float(selected[:, :kk].min(1).mean())
        result[f"region_recall@{k}"] = float(np.mean(np.any(selected[:, :kk] <= epsilon, axis=1)))
    return result


@torch.no_grad()
def evaluate_retrieval(
    args: argparse.Namespace,
    data: dict,
    bank: dict,
    model: RegionLocator,
    policy: nn.Module,
    matcher,
    scale: torch.Tensor,
    val_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_sources: torch.Tensor,
    epsilon: float,
    device: str,
    output: Path,
) -> dict:
    bank_features = bank["condition_feature"].float()
    bank_keys = model.encode_key(bank_features.to(device))
    feature_keys = _normalise_rows(bank_features)
    use_proprio = bool(bank["proprio_available"]) and bank["proprio"].shape[-1] > 0 and "proprio" in data
    prop_keys = _normalise_rows(bank["proprio"]) if use_proprio else None
    q_features = data["context"][val_ids].flatten(1).float()
    q_embed = model.encode_query(q_features.to(device))
    all_scores = (q_embed @ bank_keys.T).cpu().numpy()
    brl_all = np.argsort(-all_scores, axis=1)[:, :8]
    obs_scores = (_normalise_rows(q_features) @ feature_keys.T).numpy()
    obs_all = np.argsort(-obs_scores, axis=1)[:, :8]
    if use_proprio:
        prop_scores = (_normalise_rows(data["proprio"][val_ids].float()) @ prop_keys.T).numpy()
        prop_all = np.argsort(-prop_scores, axis=1)[:, :8]
    else:
        prop_all = obs_all.copy()
    rng = np.random.default_rng(args.seed + 59)
    random_all = np.stack([
        rng.choice(len(bank["z_star"]), size=8, replace=len(bank["z_star"]) < 8)
        for _ in range(len(val_ids))
    ])

    # The candidate-pool diagnostic is the expensive, behavior-labeled oracle.
    pool_errors = decode_pair_errors(
        policy, matcher, data, bank, val_ids, candidate_ids, scale,
        args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    brl_errors = decode_pair_errors(
        policy, matcher, data, bank, val_ids, torch.as_tensor(brl_all),
        scale, args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    obs_errors = decode_pair_errors(
        policy, matcher, data, bank, val_ids, torch.as_tensor(obs_all),
        scale, args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    prop_errors = decode_pair_errors(
        policy, matcher, data, bank, val_ids, torch.as_tensor(prop_all),
        scale, args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    random_errors = decode_pair_errors(
        policy, matcher, data, bank, val_ids, torch.as_tensor(random_all),
        scale, args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    z_global = torch.randn((len(val_ids), 1, *data["z_star"].shape[1:]), generator=torch.Generator().manual_seed(args.seed + 61))
    global_errors = decode_direct_errors(
        policy, matcher, data, val_ids, z_global, scale,
        args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )[:, 0]
    inversion_errors = decode_direct_errors(
        policy, matcher, data, val_ids, data["z_star"][val_ids, None], scale,
        args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )[:, 0]

    # Shuffled context is a control; current condition and expert target remain paired.
    permutation = torch.as_tensor(rng.permutation(len(val_ids)), dtype=torch.long)
    shuffled_features = q_features[permutation]
    shuffled_scores = (model.encode_query(shuffled_features.to(device)) @ bank_keys.T).cpu().numpy()
    shuffled_all = np.argsort(-shuffled_scores, axis=1)[:, :8]
    shuffled_errors = decode_pair_errors(
        policy, matcher, data, bank, val_ids, torch.as_tensor(shuffled_all),
        scale, args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )

    metrics = {
        "inversion_lower_bound": {"mean": float(inversion_errors.mean())},
        "global_gaussian_single": {"mean": float(global_errors.mean())},
        "candidate_pool_oracle": {"mean": float(pool_errors.min(1).mean()), "p90": float(np.quantile(pool_errors.min(1), .9))},
        "observation_retrieval": _method_summary(obs_errors, np.arange(8)[None].repeat(len(val_ids), axis=0), epsilon),
        "proprio_retrieval": _method_summary(prop_errors, np.arange(8)[None].repeat(len(val_ids), axis=0), epsilon),
        "random_train_bank": _method_summary(random_errors, np.arange(8)[None].repeat(len(val_ids), axis=0), epsilon),
        "brl": _method_summary(brl_errors, np.arange(8)[None].repeat(len(val_ids), axis=0), epsilon),
        "shuffled_context_brl": _method_summary(shuffled_errors, np.arange(8)[None].repeat(len(val_ids), axis=0), epsilon),
        "candidate_pool_oracle_region_hit": {
            "hit@1": float(np.mean(pool_errors[:, :1].min(1) <= epsilon)),
            "recall@4": float(np.mean(np.any(pool_errors[:, :4] <= epsilon, axis=1))),
            "recall@8": float(np.mean(np.any(pool_errors[:, :8] <= epsilon, axis=1))),
        },
    }
    rows = []
    for i, qid in enumerate(val_ids.tolist()):
        row = {
            "sample_index": int(qid),
            "episode_id": int(data["episode"][qid]),
            "inversion_error": float(inversion_errors[i]),
            "global_gaussian_error": float(global_errors[i]),
            "candidate_pool_oracle_error": float(pool_errors[i].min()),
            "brl_top1_error": float(brl_errors[i, 0]),
            "brl_best4_error": float(brl_errors[i, :4].min()),
            "brl_best8_error": float(brl_errors[i, :8].min()),
            "observation_top1_error": float(obs_errors[i, 0]),
            "proprio_top1_error": float(prop_errors[i, 0]),
            "random_bank_top1_error": float(random_errors[i, 0]),
            "shuffled_brl_top1_error": float(shuffled_errors[i, 0]),
            "brl_hit1": int(brl_errors[i, 0] <= epsilon),
            "brl_hit4": int(np.any(brl_errors[i, :4] <= epsilon)),
            "brl_hit8": int(np.any(brl_errors[i, :8] <= epsilon)),
        }
        rows.append(row)
    write_rows(output / "val_retrieval.csv", rows)
    atomic_save({
        "val_indices": val_ids,
        "candidate_indices": candidate_ids,
        "candidate_sources": candidate_sources,
        "candidate_errors": torch.from_numpy(pool_errors),
        "brl_top_indices": torch.from_numpy(brl_all),
        "brl_errors": torch.from_numpy(brl_errors),
        "observation_top_indices": torch.from_numpy(obs_all),
        "proprio_top_indices": torch.from_numpy(prop_all),
        "shuffled_top_indices": torch.from_numpy(shuffled_all),
        "shuffled_errors": torch.from_numpy(shuffled_errors),
        "epsilon_region": epsilon,
    }, output / "retrieval_results.pt")
    write_json(metrics, output / "metrics.json")
    return metrics


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache)
    policy, matcher, _ = load_flow(args)
    check_settings(data, manifest, policy, args)
    before = digest(policy)
    bank = build_bank(data, output)
    train_ids = torch.where(train_mask(data))[0]
    val_ids = torch.where(validation_mask(data))[0]
    if args.max_train_queries:
        train_ids = train_ids[:args.max_train_queries]
    if args.max_val_samples:
        val_ids = val_ids[:args.max_val_samples]
    train_candidates, train_sources = build_candidate_pool(data, bank, train_ids, args)
    val_candidates, val_sources = build_candidate_pool(data, bank, val_ids, args)
    atomic_save({"candidate_indices": train_candidates, "candidate_sources": train_sources, "query_indices": train_ids}, output / "train_candidates.pt")
    atomic_save({"candidate_indices": val_candidates, "candidate_sources": val_sources, "query_indices": val_ids}, output / "val_candidates.pt")

    # Ranking validation is always performed against the 200-step teacher.
    ranking_n = min(len(train_ids), max(1, math.ceil(args.ranking_pairs / train_candidates.shape[1])))
    ranking_ids = train_ids[:ranking_n]
    ranking_candidates = train_candidates[:ranking_n]
    reference_ranking = decode_pair_errors(
        policy, matcher, data, bank, ranking_ids, ranking_candidates, action_scale(data),
        args.reference_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    label_steps = args.teacher_label_steps
    label_ranking = decode_pair_errors(
        policy, matcher, data, bank, ranking_ids, ranking_candidates, action_scale(data),
        label_steps, device, args.query_batch_size, args.flow_batch_size,
    )
    ranking_report = {str(label_steps): teacher_ranking_report(label_ranking, reference_ranking, args.ranking_pairs)}
    if (not np.isfinite(ranking_report[str(label_steps)]["spearman"]) or
            ranking_report[str(label_steps)]["spearman"] < args.teacher_min_spearman) and label_steps != args.teacher_fallback_steps:
        label_steps = args.teacher_fallback_steps
        label_ranking = decode_pair_errors(
            policy, matcher, data, bank, ranking_ids, ranking_candidates, action_scale(data),
            label_steps, device, args.query_batch_size, args.flow_batch_size,
        )
        ranking_report[str(label_steps)] = teacher_ranking_report(label_ranking, reference_ranking, args.ranking_pairs)
    if (not np.isfinite(ranking_report[str(label_steps)]["spearman"]) or
            ranking_report[str(label_steps)]["spearman"] < args.teacher_min_spearman) and label_steps != args.reference_steps:
        label_steps = args.reference_steps
        ranking_report[str(label_steps)] = teacher_ranking_report(
            reference_ranking, reference_ranking, args.ranking_pairs
        )
    if label_steps == args.reference_steps:
        train_teacher = decode_pair_errors(
            policy, matcher, data, bank, train_ids, train_candidates, action_scale(data),
            label_steps, device, args.query_batch_size, args.flow_batch_size,
        )
    else:
        train_teacher = decode_pair_errors(
            policy, matcher, data, bank, train_ids, train_candidates, action_scale(data),
            label_steps, device, args.query_batch_size, args.flow_batch_size,
        )
    write_json({
        "label_steps_initial": args.teacher_label_steps,
        "label_steps_selected": label_steps,
        "reference_steps": args.reference_steps,
        "min_spearman": args.teacher_min_spearman,
        "reports": ranking_report,
    }, output / "teacher_ranking.json")
    np.savez_compressed(output / "train_teacher_errors.npz", errors=train_teacher)

    epsilon, epsilon_stats = estimate_region_epsilon(
        args, data, policy, matcher, train_ids, action_scale(data), device,
    )
    write_json({"epsilon_region": epsilon, "local_probe": epsilon_stats}, output / "region_threshold.json")

    model, history = train_locator(
        args, data, bank, train_ids, train_candidates, train_teacher, device, output,
    )
    metrics = evaluate_retrieval(
        args, data, bank, model, policy, matcher, action_scale(data), val_ids,
        val_candidates, val_sources, epsilon, device, output,
    )
    after = digest(policy)
    report = {
        "status": "ok",
        "seed": args.seed,
        "train_bank_size": len(bank["z_star"]),
        "train_queries": len(train_ids),
        "validation_queries": len(val_ids),
        "candidate_pool": int(train_candidates.shape[1]),
        "label_steps_selected": label_steps,
        "reference_steps": args.reference_steps,
        "epsilon_region": epsilon,
        "teacher_ranking": ranking_report,
        "metrics": metrics,
        "locator_final_loss": history[-1]["loss"] if history else None,
        "flow_checkpoint_sha256": sha256(args.checkpoint),
        "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"),
        "flow_parameter_digest_before": before,
        "flow_parameter_digest_after": after,
        "flow_unchanged": before == after,
        "all_flow_parameter_grads_none": all(p.grad is None for p in policy.parameters()),
        "proprio_available": bool(bank["proprio_available"]),
        "leakage_checks": {
            "bank_split": "train_only",
            "future_observation": True,
            "validation_latents_in_bank": False,
            "validation_actions_used_only_for_offline_diagnostic": True,
            "phase_used_as_locator_input": False,
        },
    }
    write_json(report, output / "stage1_report.json")
    write_json({"args": vars(args), "report_file": "stage1_report.json"}, output / "config.json")
    print(json.dumps({
        "teacher_label_steps": label_steps,
        "teacher_spearman": ranking_report[str(label_steps)]["spearman"],
        "epsilon_region": epsilon,
        "brl_top1": metrics["brl"]["top1_mean"],
        "observation_top1": metrics["observation_retrieval"]["top1_mean"],
        "global_gaussian": metrics["global_gaussian_single"]["mean"],
        "flow_unchanged": before == after,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
