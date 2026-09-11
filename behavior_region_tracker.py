"""Track a behavior-specific latent region from the previously executed chunk.

This is the successor to observation-only region retrieval.  At an online
policy update the previous executed action chunk is inverted under its own
condition and becomes the anchor ``z_prev``.  A small residual model receives
``(z_prev, c_prev, c_now)`` and predicts a local correction.  It never sees a
future observation, an expert action, or a validation latent.

The command line program first trains a staleness gate from train-episode
reuse errors, then fits the residual only on the train pairs marked stale;
it evaluates the two-stage policy offline on validation pairs.  The evaluation
also reports the decisive continuity baselines:

* current inversion (offline lower bound),
* previous-latent reuse (track-without-correction),
* global Gaussian source,
* old observation nearest-neighbour retrieval, and
* the learned local tracker.

The default policy is deliberately a no-op correction:

    invert -> reuse -> detect staleness -> correct only when needed

The residual and gate are opt-in with ``--enable-correction``.  Geometry is
applied after gate multiplication, so a reliable previous region is reused
exactly and ``tangent_norm`` cannot make a destructive radial move.  The Flow
is frozen throughout.
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
)
from flow_latent_predictor_common import (
    atomic_save,
    load_cache,
    load_flow,
    reverse_flow,
    seed_all,
    sha256,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--forward-steps", type=int, default=200)
    parser.add_argument("--reverse-steps", type=int, default=200)
    parser.add_argument("--flow-batch-size", type=int, default=32)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-step-rms", type=float, default=0.25)
    parser.add_argument(
        "--geometry",
        choices=("reuse", "tracker", "tangent", "tangent_norm"),
        default="reuse",
        help="local update used after the staleness gate fires",
    )
    parser.add_argument(
        "--enable-correction",
        action="store_true",
        help="train/use the residual and staleness gate; default is safe previous-region reuse",
    )
    parser.add_argument(
        "--staleness-quantile",
        type=float,
        default=0.95,
        help="train-only quantile of normal reuse error used as the stale threshold",
    )
    parser.add_argument("--staleness-temperature", type=float, default=0.01)
    parser.add_argument(
        "--min-correction-pairs",
        type=int,
        default=16,
        help="minimum train-only stale pairs required before fitting a residual",
    )
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--global-samples", type=int, default=1)
    parser.add_argument(
        "--save-bank",
        action="store_true",
        help="save a train-only bank for the observation-retrieval baseline",
    )
    parser.add_argument(
        "--verify-executed-inversion",
        type=int,
        default=0,
        help="re-invert this many cached previous action chunks as an online API smoke check",
    )
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_adjacent_pairs(data: dict, split_value: int) -> torch.Tensor:
    """Return [previous,current] rows with no episode or future leakage."""
    rows = []
    for episode in torch.unique(data["episode"]):
        episode_rows = torch.where(
            (data["episode"] == episode) & (data["split"] == split_value)
        )[0]
        if len(episode_rows) < 2:
            continue
        ordered = episode_rows[torch.argsort(data["condition_id"][episode_rows])]
        previous, current = ordered[:-1], ordered[1:]
        adjacent = (
            (data["condition_id"][current] - data["condition_id"][previous] == 1)
            & (data["sampler_index"][current] - data["sampler_index"][previous] == 1)
            & (data["observation_indices"][previous, -1] <= data["condition_id"][previous])
            & (data["observation_indices"][current, -1] <= data["condition_id"][current])
        )
        rows.extend(torch.stack([previous[adjacent], current[adjacent]], dim=1).tolist())
    if not rows:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def context_summary(context: torch.Tensor) -> torch.Tensor:
    """Compress the causal encoded history without using a visual backbone."""
    if context.ndim != 3 or context.shape[1] < 8:
        raise ValueError(f"expected [B,12,C] context, got {tuple(context.shape)}")
    recent = context[:, -8:].float()
    return torch.cat((recent[:, -1], recent.mean(1)), dim=-1)


def build_tracker_features(
    z_prev: torch.Tensor,
    previous_context: torch.Tensor,
    current_context: torch.Tensor,
) -> torch.Tensor:
    """Build the causal tracker input used both offline and online."""
    if z_prev.ndim != 3:
        raise ValueError(f"expected previous latent [B,16,9], got {tuple(z_prev.shape)}")
    return torch.cat(
        (
            z_prev.float().flatten(1),
            context_summary(previous_context),
            context_summary(current_context),
        ),
        dim=-1,
    )


def tracker_features(data: dict, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return build_tracker_features(
        data["z_star"][previous],
        data["context"][previous],
        data["context"][current],
    )


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).square().mean(-1).sqrt()


def geometry_update(
    z_prev: torch.Tensor,
    delta: torch.Tensor,
    variant: str,
    max_step_rms: float,
) -> torch.Tensor:
    """Apply a bounded local correction around the previous latent."""
    if variant == "reuse":
        return z_prev.clone()
    if delta.shape != z_prev.shape:
        delta = delta.reshape_as(z_prev)
    if variant in ("tangent", "tangent_norm"):
        flat_z = z_prev.flatten(1)
        flat_delta = delta.flatten(1)
        denom = flat_z.square().sum(-1, keepdim=True).clamp_min(1e-12)
        flat_delta = flat_delta - (flat_delta * flat_z).sum(-1, keepdim=True) / denom * flat_z
        delta = flat_delta.reshape_as(z_prev)
    step = _rms(delta).clamp_min(1e-12)
    multiplier = torch.minimum(
        torch.ones_like(step),
        torch.as_tensor(float(max_step_rms), device=step.device) / step,
    )
    candidate = z_prev + delta * multiplier.view(-1, 1, 1)
    if variant == "tangent_norm":
        old_norm = z_prev.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        new_norm = candidate.flatten(1).norm(dim=-1, keepdim=True).clamp_min(1e-12)
        candidate = candidate * (old_norm / new_norm).view(-1, 1, 1)
    return candidate


class TemporalRegionTracker(nn.Module):
    """Predict a residual movement from the previously selected region."""

    def __init__(self, input_dim: int, output_shape: tuple[int, int], hidden_dim: int,
                 input_mean: torch.Tensor, input_std: torch.Tensor,
                 target_mean: torch.Tensor, target_std: torch.Tensor):
        super().__init__()
        self.output_shape = tuple(output_shape)
        self.register_buffer("input_mean", input_mean.float())
        self.register_buffer("input_std", input_std.float().clamp_min(1e-5))
        self.register_buffer("target_mean", target_mean.float())
        self.register_buffer("target_std", target_std.float().clamp_min(1e-5))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, int(np.prod(output_shape))),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = (features.float() - self.input_mean) / self.input_std
        y = self.net(x) * self.target_std + self.target_mean
        return y.reshape(-1, *self.output_shape)


class StalenessGate(nn.Module):
    """Predict whether the previous region is still reliable.

    The gate is trained against a train-only reuse-error teacher.  At
    inference it receives only causal context and the previous source latent;
    the expert action is never an input.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 input_mean: torch.Tensor, input_std: torch.Tensor):
        super().__init__()
        self.register_buffer("input_mean", input_mean.float())
        self.register_buffer("input_std", input_std.float().clamp_min(1e-5))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = (features.float() - self.input_mean) / self.input_std
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_tracker(path: str | Path, device: str = "cpu") -> TemporalRegionTracker:
    """Load a trained residual tracker without touching the Flow checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    model = TemporalRegionTracker(
        int(payload["input_dim"]),
        tuple(payload["output_shape"]),
        int(payload["hidden_dim"]),
        state["input_mean"], state["input_std"],
        state["target_mean"], state["target_std"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_gate(path: str | Path, device: str = "cpu") -> StalenessGate:
    """Load a train-only staleness gate."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    model = StalenessGate(
        int(payload["input_dim"]), int(payload["hidden_dim"]),
        state["input_mean"], state["input_std"],
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def invert_executed_chunk(policy, matcher, action_raw: torch.Tensor,
                          previous_condition: torch.Tensor, reverse_steps: int) -> torch.Tensor:
    """Convert the causal executed action chunk into a source latent."""
    action_raw = action_raw.float()
    normalized = policy.normalizer["action"].normalize(action_raw.to(previous_condition.device))
    return reverse_flow(policy, matcher, normalized, previous_condition, reverse_steps)


@torch.no_grad()
def track_from_executed_chunk(
    policy,
    matcher,
    tracker: TemporalRegionTracker | None,
    gate: StalenessGate | None,
    action_raw: torch.Tensor,
    previous_condition: torch.Tensor,
    previous_context: torch.Tensor,
    current_context: torch.Tensor,
    reverse_steps: int = 200,
    geometry: str = "reuse",
    max_step_rms: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online successor: invert, reuse, and correct only when stale.

    Returns ``(z_prev, z_current_init, gate)``.  With no trained tracker/gate
    this function is exactly previous-region reuse.  ``action_raw`` is the
    complete ``[B,16,9]`` chunk used by the policy execution wrapper; no
    expert action or future observation is required.
    """
    z_prev = invert_executed_chunk(
        policy, matcher, action_raw, previous_condition, reverse_steps
    )
    if tracker is None or gate is None:
        return z_prev, z_prev.clone(), torch.zeros(z_prev.shape[0], device=z_prev.device)
    features = build_tracker_features(z_prev, previous_context, current_context)
    tracker_device = next(tracker.parameters()).device
    features_device = features.to(tracker_device)
    delta = tracker(features_device).to(z_prev.device)
    gate_value = gate(features_device).to(z_prev.device)
    z_init = geometry_update(
        z_prev, delta * gate_value.view(-1, 1, 1), geometry, max_step_rms
    )
    return z_prev, z_init, gate_value


def paired_ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float(values.mean()), float(values.mean())]
    half = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    return [float(values.mean() - half), float(values.mean() + half)]


def train_tracker(
    args: argparse.Namespace,
    data: dict,
    pairs: torch.Tensor,
    output: Path,
    device: str,
) -> tuple[TemporalRegionTracker, list[dict]]:
    previous, current = pairs[:, 0], pairs[:, 1]
    features = tracker_features(data, previous, current)
    targets = (data["z_star"][current] - data["z_star"][previous]).float()
    target_flat = targets.flatten(1)
    model = TemporalRegionTracker(
        features.shape[-1], tuple(targets.shape[1:]), args.hidden_dim,
        features.mean(0), features.std(0, unbiased=False),
        target_flat.mean(0), target_flat.std(0, unbiased=False),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 71)
    history = []
    for epoch in range(args.train_epochs):
        order = torch.randperm(len(pairs), generator=generator)
        losses = []
        for begin in range(0, len(order), args.batch_size):
            ids = order[begin:begin + args.batch_size]
            prediction = model(features[ids].to(device))
            loss = F.smooth_l1_loss(prediction, targets[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        history.append(row)
        print(json.dumps(row), flush=True)
    atomic_save({
        "state_dict": model.state_dict(),
        "input_dim": int(features.shape[-1]),
        "output_shape": list(targets.shape[1:]),
        "hidden_dim": args.hidden_dim,
        "train_pairs": pairs,
    }, output / "tracker.pt")
    write_rows(output / "tracker_training.csv", history)
    return model, history


def train_staleness_gate(
    args: argparse.Namespace,
    data: dict,
    policy: nn.Module,
    matcher,
    pairs: torch.Tensor,
    scale: torch.Tensor,
    device: str,
    output: Path,
) -> tuple[StalenessGate, dict, list[dict], torch.Tensor]:
    """Train a gate from train-only action-space reuse errors.

    The teacher is deliberately not latent distance.  A pair is stale when
    decoding ``z_prev`` under the current condition produces unusually large
    executed-action error.  The expert action is used only to construct this
    offline train label.
    """
    previous, current = pairs[:, 0], pairs[:, 1]
    features = tracker_features(data, previous, current)
    z_prev = data["z_star"][previous].float()[:, None]
    reuse_error = decode_direct_errors(
        policy, matcher, data, current, z_prev, scale,
        args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
    )[:, 0]
    threshold = float(np.quantile(reuse_error, args.staleness_quantile))
    temperature = max(float(args.staleness_temperature), 1e-6)
    targets = 1.0 / (1.0 + np.exp(-(reuse_error - threshold) / temperature))
    model = StalenessGate(
        features.shape[-1], args.hidden_dim,
        features.mean(0), features.std(0, unbiased=False),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 83)
    history = []
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    for epoch in range(args.train_epochs):
        order = torch.randperm(len(pairs), generator=generator)
        losses = []
        for begin in range(0, len(order), args.batch_size):
            ids = order[begin:begin + args.batch_size]
            prediction = model(features[ids].to(device))
            loss = F.mse_loss(prediction, target_tensor[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        history.append(row)
        print(json.dumps({"gate": row}), flush=True)
    atomic_save({
        "state_dict": model.state_dict(),
        "input_dim": int(features.shape[-1]),
        "hidden_dim": args.hidden_dim,
        "staleness_threshold": threshold,
        "staleness_quantile": args.staleness_quantile,
        "staleness_temperature": temperature,
        "train_reuse_error_mean": float(reuse_error.mean()),
        "train_reuse_error_p95": float(np.quantile(reuse_error, .95)),
    }, output / "gate.pt")
    write_rows(output / "gate_training.csv", history)
    info = {
        "threshold": threshold,
        "quantile": args.staleness_quantile,
        "temperature": temperature,
        "reuse_error_mean": float(reuse_error.mean()),
        "reuse_error_p95": float(np.quantile(reuse_error, .95)),
        "teacher_gate_positive_fraction": float(np.mean(targets > 0.5)),
    }
    return model, info, history, torch.as_tensor(targets > 0.5, dtype=torch.bool)


@torch.no_grad()
def verify_executed_inversion(args, data, policy, matcher, pairs, device) -> dict:
    count = min(args.verify_executed_inversion, len(pairs))
    if count <= 0:
        return {"requested": 0}
    previous = pairs[:count, 0]
    action = data["expert_raw"][previous].float().to(device)
    condition = data["condition"][previous].float().to(device)
    z = invert_executed_chunk(policy, matcher, action, condition, args.reverse_steps)
    cached = data["z_star"][previous].float().to(device)
    rms = _rms(z - cached).cpu().numpy()
    return {
        "requested": count,
        "rms_mean": float(rms.mean()),
        "rms_max": float(rms.max()),
        "reverse_steps": args.reverse_steps,
    }


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    data: dict,
    policy: nn.Module,
    matcher,
    tracker: TemporalRegionTracker | None,
    gate: StalenessGate | None,
    pairs: torch.Tensor,
    scale: torch.Tensor,
    device: str,
    bank: dict | None,
) -> tuple[list[dict], dict]:
    previous, current = pairs[:, 0], pairs[:, 1]
    z_prev = data["z_star"][previous].float()
    z_current = data["z_star"][current].float()
    if tracker is None or gate is None:
        predicted_delta = torch.zeros_like(z_prev)
        gate_value = torch.zeros(len(pairs), dtype=torch.float32)
    else:
        features = tracker_features(data, previous, current)
        predicted_delta_parts = []
        gate_parts = []
        tracker.eval()
        gate.eval()
        tracker_device = next(tracker.parameters()).device
        for begin in range(0, len(pairs), args.batch_size):
            feature_batch = features[begin:begin + args.batch_size].to(tracker_device)
            predicted_delta_parts.append(tracker(feature_batch).cpu())
            gate_parts.append(gate(feature_batch).cpu())
        predicted_delta = torch.cat(predicted_delta_parts)
        gate_value = torch.cat(gate_parts)
    gated_delta = predicted_delta * gate_value.view(-1, 1, 1)
    z_gated = geometry_update(z_prev, gated_delta, args.geometry, args.max_step_rms)
    generator = torch.Generator().manual_seed(args.seed + 109)
    z_gaussian = torch.randn(
        (len(pairs), args.global_samples, *z_prev.shape[1:]), generator=generator
    )
    errors = {}
    for name, z in (
        ("current", z_current[:, None]),
        ("reuse", z_prev[:, None]),
        ("gated", z_gated[:, None]),
        ("gaussian", z_gaussian),
    ):
        errors[name] = decode_direct_errors(
            policy, matcher, data, current, z, scale,
            args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
        )
    if bank is not None:
        bank_feature = F.normalize(bank["condition_feature"].float(), dim=-1)
        current_feature = F.normalize(data["context"][current].flatten(1).float(), dim=-1)
        nearest = (current_feature @ bank_feature.T).argmax(-1)
        errors["observation_retrieval"] = decode_pair_errors(
            policy, matcher, data, bank, current, nearest[:, None], scale,
            args.forward_steps, device, args.query_batch_size, args.flow_batch_size,
        )
    rows = []
    for i, (prev_id, curr_id) in enumerate(pairs.tolist()):
        row = {
            "previous_sample_index": int(prev_id),
            "current_sample_index": int(curr_id),
            "episode_id": int(data["episode"][curr_id]),
            "previous_timestep": int(data["condition_id"][prev_id]),
            "current_timestep": int(data["condition_id"][curr_id]),
            "latent_step_rms": float(_rms(z_current[i:i + 1] - z_prev[i:i + 1])[0]),
            "predicted_step_rms": float(_rms(predicted_delta[i:i + 1])[0]),
            "gate": float(gate_value[i]),
            "gated_step_rms": float(_rms(z_gated[i:i + 1] - z_prev[i:i + 1])[0]),
            "latent_norm_prev": float(z_prev[i].flatten().norm()),
            "latent_norm_gated": float(z_gated[i].flatten().norm()),
        }
        for name, values in errors.items():
            row[f"e_{name}"] = float(values[i].mean())
            if name == "gaussian":
                row["e_gaussian_best"] = float(values[i].min())
        rows.append(row)
    summary = {}
    for name in errors:
        values = np.asarray([r[f"e_{name}"] for r in rows], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)),
            "ci95": paired_ci(values),
        }
    for name in ("reuse", "gated", "observation_retrieval"):
        if name in summary:
            values = np.asarray([r[f"e_{name}"] for r in rows])
            gaussian = np.asarray([r["e_gaussian"] for r in rows])
            current_values = np.asarray([r["e_current"] for r in rows])
            summary[name]["fraction_better_than_gaussian"] = float(np.mean(values < gaussian))
            summary[name]["fraction_better_than_current"] = float(np.mean(values < current_values))
    summary["n_pairs"] = len(rows)
    summary["geometry"] = args.geometry
    summary["gate_mean"] = float(gate_value.mean())
    summary["gate_positive_rate"] = float(np.mean(gate_value.numpy() > 0.5))
    return rows, summary


def main() -> None:
    args = parse_args()
    if args.forward_steps != 200 or args.reverse_steps != 200:
        raise ValueError("formal tracking evaluation requires 200-step Flow in both directions")
    seed_all(args.seed)
    device = str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data, manifest = load_cache(args.cache)
    policy, matcher, _ = load_flow(args)
    check_settings(data, manifest, policy, args)
    before = digest(policy)
    train_pairs = make_adjacent_pairs(data, 0)
    val_pairs = make_adjacent_pairs(data, 1)
    if args.max_train_pairs:
        train_pairs = train_pairs[:args.max_train_pairs]
    if args.max_val_pairs:
        val_pairs = val_pairs[:args.max_val_pairs]
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise ValueError(f"need train and validation adjacent pairs, got {len(train_pairs)} / {len(val_pairs)}")
    scale = action_scale(data)
    tracker = None
    gate = None
    tracker_history = []
    gate_history = []
    gate_info = {"enabled": False}
    correction_train_pairs = 0
    if args.enable_correction:
        gate, gate_info, gate_history, stale_mask = train_staleness_gate(
            args, data, policy, matcher, train_pairs, scale, device, output
        )
        gate_info["enabled"] = True
        stale_pairs = train_pairs[stale_mask]
        correction_train_pairs = len(stale_pairs)
        gate_info["stale_pair_count"] = correction_train_pairs
        if correction_train_pairs >= args.min_correction_pairs:
            tracker, tracker_history = train_tracker(
                args, data, stale_pairs, output, device
            )
            gate_info["correction_fitted_on"] = "stale train pairs only"
        else:
            gate_info["correction_fitted_on"] = "none; insufficient stale train pairs"
    bank = build_bank(data, output) if args.save_bank else None
    inversion_check = verify_executed_inversion(args, data, policy, matcher, val_pairs, device)
    train_rows, train_summary = evaluate(
        args, data, policy, matcher, tracker, gate, train_pairs, scale, device, None
    )
    val_rows, val_summary = evaluate(
        args, data, policy, matcher, tracker, gate, val_pairs, scale, device, bank
    )
    write_rows(output / "tracker_train.csv", train_rows)
    write_rows(output / "tracker_val.csv", val_rows)
    after = digest(policy)
    report = {
        "status": "ok",
        "method": "Temporal Behavior Region Tracking",
        "correction_enabled": bool(args.enable_correction),
        "train": train_summary,
        "val": val_summary,
        "staleness_gate": gate_info,
        "geometry": {
            "selected": args.geometry,
            "max_step_rms": args.max_step_rms,
            "decision_rule": "reuse when gate is low; apply local correction only when gate is high",
            "variants_reported": ["current", "reuse", "gated", "gaussian"],
        },
        "pairs": {"train": len(train_pairs), "val": len(val_pairs)},
        "correction_train_pairs": correction_train_pairs,
        "executed_action_inversion_check": inversion_check,
        "flow_steps": {"forward": args.forward_steps, "reverse": args.reverse_steps},
        "flow_checkpoint_sha256": sha256(args.checkpoint),
        "cache_manifest_sha256": sha256(Path(args.cache) / "manifest.json"),
        "flow_parameter_digest_before": before,
        "flow_parameter_digest_after": after,
        "flow_unchanged": before == after,
        "all_flow_parameter_grads_none": all(p.grad is None for p in policy.parameters()),
        "leakage_checks": {
            "tracker_training_pairs": "train episodes only",
            "gate_training_labels": "train-only reuse action errors",
            "bank": "train episodes only when --save-bank is used",
            "previous_action": "causal previous chunk only",
            "current_expert_action": "offline labels/errors only",
            "future_observation": False,
            "phase_input_or_supervision": False,
        },
        "interpretation": {
            "tracking_support": "reuse remains below global Gaussian on adjacent normal transitions",
            "staleness_test": "evaluate reuse error under 0/1/2/3/5 cm target relocation before enabling correction",
            "relocalization_trigger": "if reuse rises after a closed-loop change point, open the gate and apply correction or refresh the anchor",
        },
    }
    write_json(report, output / "tracker_report.json")
    write_json({"args": vars(args), "report_file": "tracker_report.json"}, output / "config.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
