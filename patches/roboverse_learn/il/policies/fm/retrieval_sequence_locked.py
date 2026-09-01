"""Phase-filtered local inverse-memory retrieval for PickCube.

The frozen checkpoint and DefaultEvalRunner are reused.  The only policy
change is how a source state is selected from an already-built inverse bank:
global/phase random, global nearest observation, phase-filtered nearest
observation, or the same-item raw action replay control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from roboverse_learn.il.policies.fm import visual_phase_inversion_skill_bank as phase_bank
from roboverse_learn.il.policies.fm.action_memory_qkv import (
    ActionMemoryQKV,
    FlowConditionActionQKV,
)
from roboverse_learn.il.policies.fm.pickcube_phase_inversion_cross_episode_clustering import (
    PHASES,
    gather_windows,
    load_pickcube_arrays,
    load_policy,
    make_normalizer,
    precompute_frame_features,
)
from roboverse_learn.il.policies.fm.visual_phase_inversion_skill_bank import (
    precompute_visual_frame_features,
)
from roboverse_learn.il.runners.default_runner import DefaultRunner


WRONG_PHASE = {0: 3, 1: 0, 2: 3, 3: 2}
RUNNER_CONFIG: dict[str, Any] | None = None
# Optional, process-local capture used only by the success-memory insertion
# diagnostic.  It is deliberately populated by the retrieval runner rather
# than DefaultEvalRunner, so the evaluation protocol stays untouched.
EPISODIC_SOURCE_CAPTURE: list[dict[str, Any]] | None = None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _load_dinov3(model_path: str, device: torch.device):
    """Load a local DINOv3 encoder without touching the IsaacSim environment."""
    # Transformers imports audio utilities while resolving the vision model.
    # The IsaacSim environment contains a mismatched optional torchaudio
    # wheel, so disable only that optional capability in-process.
    import transformers
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    import_utils.is_torchaudio_available = lambda: False
    transformers_utils.is_torchaudio_available = lambda: False
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def _dinov3_features(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Extract normalized global DINOv3 features from CHW or HWC RGB images."""
    images = images.to(device=device, dtype=torch.float32)
    if images.ndim != 4:
        raise ValueError(f"Expected image batch [B,C,H,W] or [B,H,W,C], got {tuple(images.shape)}")
    if images.shape[-1] == 3 and images.shape[1] != 3:
        images = images.permute(0, 3, 1, 2)
    if images.shape[1] != 3:
        raise ValueError(f"Expected RGB images, got {tuple(images.shape)}")
    if float(images.detach().amax()) > 1.5:
        images = images / 255.0
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    outputs = model(pixel_values=(images - mean) / std)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs.last_hidden_state[:, 0]
    return F.normalize(pooled.float(), dim=-1).cpu()


def build_multimodal_memory(args: argparse.Namespace) -> dict[str, Any]:
    """Attach DINOv3 image and normalized proprioception keys to x0.8 memory."""
    base_path = Path(args.memory_path).expanduser().resolve()
    output_path = Path(args.multimodal_memory_path).expanduser().resolve()
    payload = torch.load(base_path, map_location="cpu")
    if "retrieval_memory" not in payload:
        raise RuntimeError(f"{base_path} is not a retrieval memory")
    root, _, _, _, _, _, _ = load_pickcube_arrays(args)
    policy, _, _, device = load_policy(args)
    normalizer = make_normalizer(root, policy)
    dino = _load_dinov3(args.dinov3_model_path, device)

    global_memory = payload["retrieval_memory"]["__global__"]
    starts = [int(value) for value in global_memory["global_start"].tolist()]
    image_indices = [start + int(args.n_obs_steps) - 1 for start in starts]
    image_batch = torch.from_numpy(
        np.stack([np.asarray(root["data/head_camera"][index]) for index in image_indices])
    )
    dino_features = []
    for begin in range(0, len(image_batch), args.dino_batch_size):
        dino_features.append(_dinov3_features(dino, image_batch[begin : begin + args.dino_batch_size], device))
    dino_features = torch.cat(dino_features, dim=0).contiguous()

    proprio_features = []
    for start in starts:
        states = torch.from_numpy(
            np.asarray(root["data/state"][start : start + int(args.n_obs_steps)])
        ).float()
        normalized = normalizer["agent_pos"].normalize(states)
        proprio_features.append(F.normalize(normalized.flatten().float(), dim=0).cpu())
    proprio_features = torch.stack(proprio_features).contiguous()
    feature_by_start = {
        start: (dino_features[index], proprio_features[index])
        for index, start in enumerate(starts)
    }

    for values in payload["retrieval_memory"].values():
        item_starts = [int(value) for value in values["global_start"].tolist()]
        values["dinov3_feature"] = torch.stack([feature_by_start[start][0] for start in item_starts])
        values["proprio_feature"] = torch.stack([feature_by_start[start][1] for start in item_starts])
    payload["retrieval_memory_meta"].update({
        "format": "multimodal_adaptive_depth_retrieval_v1",
        "image_encoder": "DINOv3 ViT-B/16 frozen",
        "image_feature": "last observation frame, ImageNet preprocessing",
        "proprio_feature": "normalized agent_pos history",
        "image_weight": float(args.image_weight),
        "proprio_weight": float(args.proprio_weight),
        "retrieval_top_m": int(args.retrieval_top_m),
        "source_depth": 0.8,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    result = {
        "status": "multimodal_memory_built",
        "memory_path": str(output_path),
        "base_memory_path": str(base_path),
        "num_items": len(starts),
        "dinov3_model_path": args.dinov3_model_path,
        "image_weight": float(args.image_weight),
        "proprio_weight": float(args.proprio_weight),
        "retrieval_top_m": int(args.retrieval_top_m),
    }
    (output_path.parent / "multimodal_memory_summary.json").write_text(json.dumps(result, indent=2))
    return result


def build_retrieval_memory(args: argparse.Namespace) -> dict[str, Any]:
    """Add per-window observation/action metadata to the prior phase bank."""
    bank_path = Path(args.base_bank_path).expanduser().resolve()
    payload = torch.load(bank_path, map_location="cpu")
    depth_key = f"{args.depth:.1f}"
    if depth_key not in payload["banks"]:
        raise KeyError(f"Depth {depth_key} is absent from {bank_path}")

    root, starts, ends, episode_ids, windows, _, _ = load_pickcube_arrays(args)
    policy, _, _, device = load_policy(args)
    normalizer = make_normalizer(root, policy)
    total_frames = int(ends[args.num_episodes - 1])
    frame_features = precompute_frame_features(
        policy, normalizer, root, total_frames, device, args.feature_batch_size
    )
    visual_frame_features = precompute_visual_frame_features(
        policy, normalizer, root, total_frames, device, args.feature_batch_size
    )
    raw_actions, _, conditions, phases, episode_indices, purities = gather_windows(
        root, normalizer, frame_features, windows, args.horizon, args.n_obs_steps
    )

    bank_mask = episode_indices < args.bank_episodes
    selected_indices = np.where(bank_mask)[0]
    bank_phases = phases[bank_mask]
    bank_conditions = conditions[bank_mask].cpu().float()
    visual_conditions = torch.stack([
        visual_frame_features[window.global_start : window.global_start + args.n_obs_steps].flatten()
        for window in windows
    ])[bank_mask].cpu().float()
    bank_raw_actions = raw_actions[bank_mask].cpu().float()
    bank_episode_indices = torch.from_numpy(episode_indices[bank_mask].astype(np.int64))
    bank_global_starts = torch.tensor(
        [windows[index].global_start for index in selected_indices], dtype=torch.long
    )
    bank_purities = torch.from_numpy(purities[bank_mask]).float()

    memory: dict[str, dict[str, torch.Tensor]] = {}
    for phase_id, phase_name in enumerate(PHASES):
        mask = bank_phases == phase_id
        saved_source = payload["banks"][depth_key][phase_name].float().cpu().contiguous()
        if len(saved_source) != int(mask.sum()):
            raise RuntimeError(
                f"Bank order mismatch for {phase_name}: saved={len(saved_source)} "
                f"metadata={int(mask.sum())}"
            )
        memory[phase_name] = {
            "source": saved_source,
            "obs_feature": bank_conditions[mask].contiguous(),
            "visual_feature": visual_conditions[mask].contiguous(),
            "raw_action": bank_raw_actions[mask].contiguous(),
            "episode_index": bank_episode_indices[mask].contiguous(),
            "global_start": bank_global_starts[mask].contiguous(),
            "purity": bank_purities[mask].contiguous(),
        }

    global_source = payload["banks"][depth_key]["__global__"].float().cpu().contiguous()
    if len(global_source) != len(bank_conditions):
        raise RuntimeError("Saved global bank and metadata have different lengths")
    memory["__global__"] = {
        "source": global_source,
        "obs_feature": bank_conditions,
        "visual_feature": visual_conditions,
        "raw_action": bank_raw_actions,
        "episode_index": bank_episode_indices,
        "global_start": bank_global_starts,
        "purity": bank_purities,
    }
    payload["retrieval_memory"] = memory
    payload["retrieval_memory_meta"] = {
        "format": "phase_filtered_local_inverse_retrieval_v1",
        "depth": args.depth,
        "metric": "cosine_similarity",
        "bank_episodes": args.bank_episodes,
        "num_windows": int(len(bank_conditions)),
        "phase_names": list(PHASES),
        "same_episode_exclusion_supported": True,
    }
    output_path = Path(args.memory_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    stats_rows = []
    for name, values in memory.items():
        norms = values["source"].flatten(1).norm(dim=-1)
        stats_rows.append({
            "bank": name,
            "num_items": int(len(values["source"])),
            "source_norm_mean": float(norms.mean()),
            "source_norm_std": float(norms.std(unbiased=False)),
            "obs_feature_dim": int(values["obs_feature"].shape[-1]),
        })
    write_csv(output_path.parent / "retrieval_memory_statistics.csv", stats_rows)
    result = {
        "status": "memory_built",
        "memory_path": str(output_path),
        "base_bank_path": str(bank_path),
        "depth": args.depth,
        "num_windows": int(len(bank_conditions)),
        "phase_counts": {name: int(len(memory[name]["source"])) for name in PHASES},
        "metric": "cosine_similarity",
    }
    (output_path.parent / "retrieval_memory_summary.json").write_text(json.dumps(result, indent=2))
    return result


class PhaseFilteredLocalRetrievalRunner(phase_bank.VisualPhaseSkillBankRunner):
    """DefaultEvalRunner adapter; core runner/cache/success logic is untouched."""

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        # Parent restores the checkpoint model, normalizer, observation deque,
        # and runner configuration.  It reads phase_bank.RUNNER_CONFIG.
        super()._init_policy(default_runner, **kwargs)
        payload = self.bank_payload
        if "retrieval_memory" not in payload:
            raise RuntimeError("Memory payload has no retrieval_memory field")
        self.memory = {}
        for name, values in payload["retrieval_memory"].items():
            self.memory[name] = {
                key: value.float() if value.is_floating_point() else value
                for key, value in values.items()
            }
            # Keep the unnormalized policy condition for the additional
            # reverse step used by adaptive-depth retrieval.  The normalized
            # copy below is only for cosine matching.
            self.memory[name]["condition"] = self.memory[name]["obs_feature"].to(self.device)
            self.memory[name]["obs_feature"] = F.normalize(
                self.memory[name]["obs_feature"].to(self.device), dim=-1
            )
            self.memory[name]["source"] = self.memory[name]["source"].to(self.device)
            self.memory[name]["raw_action"] = self.memory[name]["raw_action"].to(self.device)
            self.memory[name]["episode_index"] = self.memory[name]["episode_index"].cpu()
            self.memory[name]["global_start"] = self.memory[name]["global_start"].cpu()
        config = RUNNER_CONFIG
        if config is None:
            raise RuntimeError("RUNNER_CONFIG is not initialized")
        self.exclude_same_episode = bool(config.get("exclude_same_episode", True))
        self.retrieval_topk = max(1, int(config.get("retrieval_topk", 1)))
        self.max_candidate_dispersion = float(config.get("max_candidate_dispersion", 0.05))
        self._previous_normalized_action = None
        self._locked_episode = [None] * self.num_envs
        self._locked_start = [None] * self.num_envs
        self._auto_similarity_threshold = self._calibrate_similarity_threshold()
        self._adaptive_similarity_thresholds = self._calibrate_adaptive_depth_thresholds()
        self.retrieval_log_path = Path(config["retrieval_log_path"])
        self.retrieval_log_path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self):
        super().reset()
        self._previous_normalized_action = None
        self._locked_episode = [None] * self.num_envs
        self._locked_start = [None] * self.num_envs

    @torch.no_grad()
    def _calibrate_similarity_threshold(self) -> float:
        """Estimate an in-support retrieval threshold from the bank itself.

        This is a label-free leave-one-episode-out calibration: sampled bank
        queries retrieve their closest item from a different episode.  The
        lower 5th percentile is used as the minimum similarity for the
        trust-gated method, so no test success labels enter the gate.
        """
        configured = RUNNER_CONFIG.get("min_similarity") if RUNNER_CONFIG else None
        if configured is not None:
            return float(configured)
        values = self.memory.get("__global__")
        if values is None or len(values["source"]) < 2:
            return -1.0
        count = min(128, len(values["source"]))
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 701)
        sample = torch.randperm(len(values["source"]), generator=generator)[:count]
        queries = values["obs_feature"][sample]
        similarities = queries @ values["obs_feature"].T
        bank_eps = values["episode_index"].to(self.device)
        for row, item in enumerate(sample.tolist()):
            similarities[row, bank_eps == bank_eps[item]] = -float("inf")
        nearest = similarities.max(dim=1).values
        finite = nearest[torch.isfinite(nearest)]
        return float(torch.quantile(finite, 0.05).item()) if len(finite) else -1.0

    @torch.no_grad()
    def _calibrate_adaptive_depth_thresholds(self) -> tuple[float, float]:
        """Get label-free low/high similarity cutoffs from LOO bank matches."""
        values = self.memory.get("__global__")
        if values is None or len(values["source"]) < 3:
            return (-1.0, 1.0)
        count = min(256, len(values["source"]))
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 1701)
        sample = torch.randperm(len(values["source"]), generator=generator)[:count]
        queries = values["obs_feature"][sample]
        similarities = queries @ values["obs_feature"].T
        bank_eps = values["episode_index"].to(self.device)
        for row, item in enumerate(sample.tolist()):
            similarities[row, bank_eps == bank_eps[item]] = -float("inf")
        nearest = similarities.max(dim=1).values
        finite = nearest[torch.isfinite(nearest)]
        if len(finite) == 0:
            return (-1.0, 1.0)
        return (
            float(torch.quantile(finite, 1.0 / 3.0).item()),
            float(torch.quantile(finite, 2.0 / 3.0).item()),
        )

    def _select_bank(self, phase_id: int) -> str:
        if self.method in {
            "global_random_inverse_bank", "global_nearest_inverse",
            "candidate_select_inverse", "candidate_gate_inverse",
            "sequence_locked_inverse", "adaptive_depth_inverse",
        }:
            return "__global__"
        if self.method in {
            "phase_random_bank", "phase_nearest_inverse", "phase_nearest_raw_replay",
            "visual_phase_nearest_inverse",
        }:
            return PHASES[phase_id]
        if self.method == "wrong_phase_nearest":
            return PHASES[WRONG_PHASE[phase_id]]
        raise ValueError(f"Unknown retrieval method: {self.method}")

    def _retrieve_candidates(self, phases: torch.Tensor, condition: torch.Tensor):
        """Retrieve real inverse states without averaging them in latent space."""
        sources, similarities, episode_ids, starts = [], [], [], []
        for env_id, phase_value in enumerate(phases.detach().cpu().reshape(-1)):
            phase_id = int(phase_value.item())
            bank_name = self._select_bank(phase_id)
            values = self.memory[bank_name]
            valid = torch.ones(len(values["source"]), dtype=torch.bool)
            query_episode = self._episode_counter * self.num_envs + env_id
            if self.exclude_same_episode:
                valid &= values["episode_index"] != query_episode
            locked_episode = self._locked_episode[env_id]
            locked_start = self._locked_start[env_id]
            if locked_episode is not None and bank_name == "__global__":
                same_locked_episode = valid & (values["episode_index"] == locked_episode)
                forward_only = same_locked_episode & (values["global_start"] > locked_start)
                if bool(forward_only.any()):
                    stride = max(1, int(self.policy.n_action_steps))
                    local_forward = forward_only & (
                        values["global_start"] <= locked_start + 4 * stride
                    )
                    valid = local_forward if bool(local_forward.any()) else forward_only
            if not bool(valid.any()):
                valid[:] = True
            query = F.normalize(condition[env_id : env_id + 1], dim=-1)
            similarities_all = (values["obs_feature"] @ query.T).squeeze(-1)
            valid_indices = torch.where(valid.to(self.device))[0]
            valid_scores = similarities_all[valid_indices]
            k = min(self.retrieval_topk, int(valid_indices.numel()))
            top_scores, order = torch.topk(valid_scores, k=k, dim=0)
            selected = valid_indices[order]
            sources.append(values["source"][selected])
            similarities.append(top_scores)
            selected_cpu = selected.detach().cpu()
            episode_ids.append(values["episode_index"][selected_cpu].clone())
            starts.append(values["global_start"][selected_cpu].clone())
            self._locked_episode[env_id] = int(values["episode_index"][selected_cpu[0]].item())
            self._locked_start[env_id] = int(values["global_start"][selected_cpu[0]].item())
        return (
            torch.stack(sources).to(self.device),
            torch.stack(similarities).to(self.device),
            episode_ids,
            starts,
        )

    def _retrieve(self, phases: torch.Tensor, condition: torch.Tensor):
        sources, raw_actions, names, item_eps, similarities, starts = [], [], [], [], [], []
        topk_eps, topk_starts, topk_weights = [], [], []
        for env_id, phase_value in enumerate(phases.detach().cpu().reshape(-1)):
            phase_id = int(phase_value.item())
            bank_name = self._select_bank(phase_id)
            values = self.memory[bank_name]
            valid = torch.ones(len(values["source"]), dtype=torch.bool)
            query_episode = self._episode_counter * self.num_envs + env_id
            if self.exclude_same_episode:
                valid &= values["episode_index"] != query_episode
            if not bool(valid.any()):
                valid[:] = True

            if self.method in {"global_random_inverse_bank", "phase_random_bank"}:
                candidates = torch.where(valid)[0]
                picked = int(torch.randint(
                    len(candidates), (1,), generator=self._source_generator
                ).item())
                index = int(candidates[picked].item())
                similarity = float("nan")
                selected = torch.tensor([index], device=self.device, dtype=torch.long)
                weights = torch.ones(1, device=self.device, dtype=values["source"].dtype)
            else:
                query = F.normalize(condition[env_id : env_id + 1], dim=-1)
                similarities_all = (values["obs_feature"] @ query.T).squeeze(-1)
                valid_indices = torch.where(valid.to(self.device))[0]
                valid_scores = similarities_all[valid_indices]
                k = min(self.retrieval_topk, int(valid_indices.numel()))
                top_scores, order = torch.topk(valid_scores, k=k, dim=0)
                selected = valid_indices[order]
                # Similarity-weighted fusion of the retrieved x0.8 states.
                # Positive cosine scores are normalized; a uniform fallback
                # keeps the source finite if a pathological bank is supplied.
                positive_scores = top_scores.clamp_min(0)
                score_sum = positive_scores.sum()
                if not bool(torch.isfinite(score_sum)) or float(score_sum) <= 1e-12:
                    weights = torch.full_like(top_scores, 1.0 / float(k))
                else:
                    weights = positive_scores / score_sum
                index = int(selected[0].item())
                similarity = float(top_scores[0].item())
            fused_source = (values["source"][selected] * weights.reshape(-1, 1, 1)).sum(dim=0)
            sources.append(fused_source)
            # Raw replay remains the top-1 paired retrieval control.
            raw_actions.append(values["raw_action"][int(selected[0].item())])
            names.append(bank_name)
            item_eps.append(int(values["episode_index"][int(selected[0].item())].item()))
            similarities.append(similarity)
            starts.append(int(values["global_start"][int(selected[0].item())].item()))
            topk_eps.append([int(values["episode_index"][item].item()) for item in selected.detach().cpu().tolist()])
            topk_starts.append([int(values["global_start"][item].item()) for item in selected.detach().cpu().tolist()])
            topk_weights.append([float(value) for value in weights.detach().cpu().tolist()])
        return (
            torch.stack(sources).to(self.device),
            torch.stack(raw_actions).to(self.device),
            names,
            item_eps,
            similarities,
            starts,
            topk_eps,
            topk_starts,
            topk_weights,
        )

    def _append_retrieval_log(self, row: dict[str, Any]) -> None:
        with self.retrieval_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    @torch.no_grad()
    def _predict_candidate_action(
        self,
        obs,
        policy,
        condition: torch.Tensor,
        phase: torch.Tensor,
        visual_phase: torch.Tensor,
        oracle_phase: torch.Tensor,
    ):
        """Forward each retrieved inverse state separately, then gate.

        The candidate states are never averaged.  A candidate is accepted only
        when its observation similarity is inside the bank's label-free
        leave-one-episode-out support threshold; otherwise the native Gaussian
        Flow path is used for that query.
        """
        sources, similarities, topk_eps, topk_starts = self._retrieve_candidates(phase, condition)
        batch_size, candidate_count = sources.shape[:2]
        flat_sources = sources.reshape(batch_size * candidate_count, *sources.shape[2:])
        flat_condition = condition.repeat_interleave(candidate_count, dim=0)
        tic = time.perf_counter()
        normalized_candidates = phase_bank.midpoint_integrate(
            policy.model,
            flat_sources,
            flat_condition,
            self.bank_depth,
            1.0,
            self.flow_steps,
        ).reshape(batch_size, candidate_count, *sources.shape[2:])
        phase_bank.sync(self._torch_device)
        candidate_forward_ms = (time.perf_counter() - tic) * 1000.0

        future_start = policy.n_obs_steps - 1
        future_end = future_start + policy.n_action_steps
        candidate_future = normalized_candidates[:, :, future_start:future_end]
        candidate_mean = candidate_future.mean(dim=1, keepdim=True)
        candidate_dispersion = (
            (candidate_future - candidate_mean).flatten(2).norm(dim=-1)
            / math.sqrt(float(candidate_future[0, 0].numel()))
        ).mean(dim=1)
        top1_similarity = similarities[:, 0]
        selected_index = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        top1_continuity = torch.full_like(top1_similarity, float("nan"))
        if self._previous_normalized_action is not None:
            previous = self._previous_normalized_action.to(self.device)
            continuity = (
                candidate_future - previous[:, None]
            ).flatten(2).norm(dim=-1) / math.sqrt(float(candidate_future[0, 0].numel()))
            top1_continuity = continuity[:, 0]

        accepted = (
            (top1_similarity >= float(self._auto_similarity_threshold))
            & (candidate_dispersion <= self.max_candidate_dispersion)
        )
        native_forward_ms = 0.0
        native_normalized_full = None
        if not bool(accepted.all()):
            native_start = time.perf_counter()
            native_result = policy.predict_action(obs)
            phase_bank.sync(self._torch_device)
            native_forward_ms = (time.perf_counter() - native_start) * 1000.0
            native_normalized_full = policy.normalizer["action"].normalize(
                native_result["action_pred"]
            ).to(torch.float32)

        chosen_normalized_full = normalized_candidates[:, 0].clone()
        if native_normalized_full is not None:
            chosen_normalized_full[~accepted] = native_normalized_full[~accepted]
        action = policy.normalizer["action"].unnormalize(chosen_normalized_full)
        self._previous_normalized_action = chosen_normalized_full[:, future_start:future_end].detach()

        for env_id in range(batch_size):
            eps = [int(value) for value in topk_eps[env_id].tolist()]
            starts = [int(value) for value in topk_starts[env_id].tolist()]
            row = {
                "episode_id": self._episode_counter * self.num_envs + env_id,
                "query_id": self._query_counter,
                "env_id": env_id,
                "env_step": self.step,
                "method": self.method,
                "phase_source": self.phase_source,
                "selected_phase": PHASES[int(phase[env_id].item())],
                "visual_phase": PHASES[int(visual_phase[env_id].item())],
                "oracle_time_phase": PHASES[int(oracle_phase[env_id].item())],
                "retrieval_topk": candidate_count,
                "retrieved_topk_episode_indices": eps,
                "retrieved_topk_global_starts": starts,
                "retrieval_similarity": float(top1_similarity[env_id].item()),
                "retrieval_distance": float(1.0 - top1_similarity[env_id].item()),
                "retrieval_threshold": float(self._auto_similarity_threshold),
                "max_candidate_dispersion": self.max_candidate_dispersion,
                "gate_accept": bool(accepted[env_id].item()),
                "fallback_to_gaussian": not bool(accepted[env_id].item()),
                "candidate_action_dispersion": float(candidate_dispersion[env_id].item()),
                "top1_continuity": float(top1_continuity[env_id].item()),
                "source_norm": float(sources[env_id, 0].flatten().norm().item()),
                "action_norm": float(chosen_normalized_full[env_id].flatten().norm().item()),
                "candidate_forward_ms": candidate_forward_ms,
                "native_forward_ms": native_forward_ms,
            }
            self._append_trace(row)
            self._append_retrieval_log(row)
        self._query_counter += 1
        return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)

    @torch.no_grad()
    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()
        policy = self.policy
        condition, visual_phase = self._encode_condition_and_visual_phase(obs)
        oracle_phase = torch.full_like(visual_phase, self._oracle_time_phase())
        phase = visual_phase if self.phase_source == "visual" else oracle_phase

        if self.method in {"candidate_select_inverse", "candidate_gate_inverse"}:
            return self._predict_candidate_action(
                obs, policy, condition, phase, visual_phase, oracle_phase
            )

        # Clean sequence-locked control: use only the top-1 real inverse
        # state and follow later windows from the same retrieved episode.
        # There is no latent averaging, trust gate, or Gaussian fallback here.
        if self.method == "sequence_locked_inverse":
            sources, similarities, topk_eps, topk_starts = self._retrieve_candidates(
                phase, condition
            )
            source = sources[:, 0]
            tic = time.perf_counter()
            normalized_trajectory = phase_bank.midpoint_integrate(
                policy.model, source, condition, self.bank_depth, 1.0, self.flow_steps
            )
            phase_bank.sync(self._torch_device)
            forward_ms = (time.perf_counter() - tic) * 1000.0
            action = policy.normalizer["action"].unnormalize(normalized_trajectory)
            future_start = policy.n_obs_steps - 1
            future_end = future_start + policy.n_action_steps
            normalized_for_log = policy.normalizer["action"].normalize(action)
            for env_id in range(condition.shape[0]):
                eps = [int(value) for value in topk_eps[env_id].tolist()]
                starts = [int(value) for value in topk_starts[env_id].tolist()]
                similarity = float(similarities[env_id, 0].item())
                row = {
                    "episode_id": self._episode_counter * self.num_envs + env_id,
                    "query_id": self._query_counter,
                    "env_id": env_id,
                    "env_step": self.step,
                    "method": self.method,
                    "phase_source": self.phase_source,
                    "selected_phase": PHASES[int(phase[env_id].item())],
                    "visual_phase": PHASES[int(visual_phase[env_id].item())],
                    "oracle_time_phase": PHASES[int(oracle_phase[env_id].item())],
                    "retrieval_topk": 1,
                    "retrieved_topk_episode_indices": eps,
                    "retrieved_topk_global_starts": starts,
                    "retrieval_similarity": similarity,
                    "retrieval_distance": 1.0 - similarity,
                    "retrieved_episode_index": eps[0],
                    "retrieved_global_start": starts[0],
                    "source_norm": float(source[env_id].flatten().norm()),
                    "action_norm": float(normalized_for_log[env_id].flatten().norm()),
                    "forward_ms": forward_ms,
                    "sequence_locked": True,
                }
                self._append_trace(row)
                self._append_retrieval_log(row)
            self._query_counter += 1
            return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)

        # Adaptive-depth retrieval: the bank stores real x0.8 states.  A
        # lower similarity query gets an additional reverse integration from
        # x0.8 to x0.6/x0.4 under the retrieved expert condition, then all
        # cases are reconditioned under the current observation.  This keeps
        # inversion as the mechanism and avoids latent averaging or fallback.
        if self.method == "adaptive_depth_inverse":
            (
                sources,
                raw_action,
                bank_names,
                item_eps,
                sims,
                starts,
                topk_eps,
                topk_starts,
                topk_weights,
            ) = self._retrieve(phase, condition)
            future_start = policy.n_obs_steps - 1
            future_end = future_start + policy.n_action_steps
            low_sim, high_sim = self._adaptive_similarity_thresholds
            generated = []
            reverse_times, forward_times, selected_depths = [], [], []
            for env_id in range(condition.shape[0]):
                similarity = float(sims[env_id])
                if similarity >= high_sim:
                    target_depth = 0.8
                    depth_bucket = "high_similarity_shallow"
                elif similarity >= low_sim:
                    target_depth = 0.6
                    depth_bucket = "mid_similarity_medium"
                else:
                    target_depth = 0.4
                    depth_bucket = "low_similarity_deep"

                source = sources[env_id : env_id + 1]
                reverse_ms = 0.0
                if target_depth < self.bank_depth:
                    values = self.memory[bank_names[env_id]]
                    matches = (
                        (values["episode_index"] == item_eps[env_id])
                        & (values["global_start"] == starts[env_id])
                    )
                    matching_indices = torch.where(matches)[0]
                    if len(matching_indices) == 0:
                        raise RuntimeError(
                            "Could not recover the retrieved expert condition for adaptive inversion"
                        )
                    index = int(matching_indices[0].item())
                    expert_condition = values["condition"][index : index + 1]
                    tic = time.perf_counter()
                    source = phase_bank.midpoint_integrate(
                        policy.model,
                        source,
                        expert_condition,
                        self.bank_depth,
                        target_depth,
                        self.flow_steps,
                    )
                    phase_bank.sync(self._torch_device)
                    reverse_ms = (time.perf_counter() - tic) * 1000.0

                tic = time.perf_counter()
                generated.append(
                    phase_bank.midpoint_integrate(
                        policy.model,
                        source,
                        condition[env_id : env_id + 1],
                        target_depth,
                        1.0,
                        self.flow_steps,
                    )
                )
                phase_bank.sync(self._torch_device)
                forward_ms = (time.perf_counter() - tic) * 1000.0
                reverse_times.append(reverse_ms)
                forward_times.append(forward_ms)
                selected_depths.append((target_depth, depth_bucket))

            normalized_trajectory = torch.cat(generated, dim=0)
            action = policy.normalizer["action"].unnormalize(normalized_trajectory)
            normalized_for_log = policy.normalizer["action"].normalize(action)
            for env_id in range(condition.shape[0]):
                eps = [int(value) for value in topk_eps[env_id]]
                starts_list = [int(value) for value in topk_starts[env_id]]
                similarity = float(sims[env_id])
                target_depth, depth_bucket = selected_depths[env_id]
                row = {
                    "episode_id": self._episode_counter * self.num_envs + env_id,
                    "query_id": self._query_counter,
                    "env_id": env_id,
                    "env_step": self.step,
                    "method": self.method,
                    "phase_source": self.phase_source,
                    "selected_phase": PHASES[int(phase[env_id].item())],
                    "visual_phase": PHASES[int(visual_phase[env_id].item())],
                    "oracle_time_phase": PHASES[int(oracle_phase[env_id].item())],
                    "retrieval_topk": self.retrieval_topk,
                    "retrieved_topk_episode_indices": eps,
                    "retrieved_topk_global_starts": starts_list,
                    "retrieval_similarity": similarity,
                    "retrieval_distance": 1.0 - similarity,
                    "retrieved_episode_index": item_eps[env_id],
                    "retrieved_global_start": starts[env_id],
                    "source_depth": self.bank_depth,
                    "adaptive_t_star": target_depth,
                    "adaptive_depth_bucket": depth_bucket,
                    "low_similarity_threshold": low_sim,
                    "high_similarity_threshold": high_sim,
                    "reverse_ms": reverse_times[env_id],
                    "forward_ms": forward_times[env_id],
                    "source_norm": float(sources[env_id].flatten().norm()),
                    "action_norm": float(normalized_for_log[env_id].flatten().norm()),
                }
                self._append_trace(row)
                self._append_retrieval_log(row)
            self._query_counter += 1
            return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)

        # Native checkpoint path is retained for the Gaussian regression.
        if self.method == "vanilla_gaussian_flow":
            result = policy.predict_action(obs)
            action_chunk = result["action"].detach().to(torch.float32).transpose(0, 1)
            normalized_full = policy.normalizer["action"].normalize(result["action_pred"])
            for env_id in range(condition.shape[0]):
                self._append_trace({
                    "episode_id": self._episode_counter * self.num_envs + env_id,
                    "query_id": self._query_counter,
                    "env_id": env_id,
                    "env_step": self.step,
                    "method": self.method,
                    "selected_phase": PHASES[int(phase[env_id].item())],
                    "selected_bank": "gaussian",
                    "retrieval_similarity": float("nan"),
                    "retrieved_episode_index": -1,
                    "retrieved_global_start": -1,
                    "source_norm": float("nan"),
                    "action_norm": float(normalized_full[env_id].flatten().norm()),
                })
            self._query_counter += 1
            return action_chunk

        source, raw_action, bank_names, item_eps, sims, starts, topk_eps, topk_starts, topk_weights = self._retrieve(phase, condition)
        if self.method == "phase_nearest_raw_replay":
            action = raw_action
            forward_ms = 0.0
        else:
            tic = time.perf_counter()
            normalized_trajectory = phase_bank.midpoint_integrate(
                policy.model, source, condition, self.bank_depth, 1.0, self.flow_steps
            )
            phase_bank.sync(self._torch_device)
            forward_ms = (time.perf_counter() - tic) * 1000.0
            action = policy.normalizer["action"].unnormalize(normalized_trajectory)

        future_start = policy.n_obs_steps - 1
        future_end = future_start + policy.n_action_steps
        normalized_for_log = policy.normalizer["action"].normalize(action)
        for env_id in range(condition.shape[0]):
            row = {
                "episode_id": self._episode_counter * self.num_envs + env_id,
                "query_id": self._query_counter,
                "env_id": env_id,
                "env_step": self.step,
                "method": self.method,
                "phase_source": self.phase_source,
                "selected_phase": PHASES[int(phase[env_id].item())],
                "visual_phase": PHASES[int(visual_phase[env_id].item())],
                "oracle_time_phase": PHASES[int(oracle_phase[env_id].item())],
                "selected_bank": bank_names[env_id],
                "retrieval_similarity": sims[env_id],
                "retrieval_distance": 1.0 - sims[env_id] if math.isfinite(sims[env_id]) else float("nan"),
                "retrieved_episode_index": item_eps[env_id],
                "retrieved_global_start": starts[env_id],
                "retrieval_topk": self.retrieval_topk,
                "retrieved_topk_episode_indices": topk_eps[env_id],
                "retrieved_topk_global_starts": topk_starts[env_id],
                "retrieved_topk_weights": topk_weights[env_id],
                "source_norm": float(source[env_id].flatten().norm()),
                "action_norm": float(normalized_for_log[env_id].flatten().norm()),
                "forward_ms": forward_ms,
            }
            self._append_trace(row)
            self._append_retrieval_log(row)
        self._query_counter += 1
        return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)


class MultimodalAdaptiveDepthRunner(PhaseFilteredLocalRetrievalRunner):
    """DINOv3-primary retrieval with proprioception reranking and adaptive t*."""

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        super()._init_policy(default_runner, **kwargs)
        config = RUNNER_CONFIG
        if config is None:
            raise RuntimeError("RUNNER_CONFIG is not initialized")
        if abs(float(self.bank_depth) - 0.8) > 1e-6:
            raise ValueError("Multimodal adaptive retrieval requires an x0.8 source bank")
        self.image_weight = float(config.get("image_weight", 0.8))
        self.proprio_weight = float(config.get("proprio_weight", 0.2))
        weight_sum = self.image_weight + self.proprio_weight
        if weight_sum <= 0:
            raise ValueError("image_weight + proprio_weight must be positive")
        self.image_weight /= weight_sum
        self.proprio_weight /= weight_sum
        self.retrieval_top_m = max(1, int(config.get("retrieval_top_m", 32)))
        for values in self.memory.values():
            values["dinov3_feature"] = F.normalize(
                values["dinov3_feature"].to(self.device), dim=-1
            )
            values["proprio_feature"] = F.normalize(
                values["proprio_feature"].to(self.device), dim=-1
            )
        self.dinov3 = _load_dinov3(config["dinov3_model_path"], self._torch_device)
        self._multimodal_depth_thresholds = self._calibrate_multimodal_depth_thresholds()

    @torch.no_grad()
    def _calibrate_multimodal_depth_thresholds(self) -> tuple[float, float]:
        values = self.memory["__global__"]
        count = min(256, len(values["source"]))
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 2701)
        sample = torch.randperm(len(values["source"]), generator=generator)[:count].to(self.device)
        image_query = values["dinov3_feature"][sample]
        proprio_query = values["proprio_feature"][sample]
        image_scores = image_query @ values["dinov3_feature"].T
        bank_eps = values["episode_index"].to(self.device)
        for row, item in enumerate(sample.tolist()):
            image_scores[row, bank_eps == bank_eps[item]] = -float("inf")
        top_count = min(self.retrieval_top_m, image_scores.shape[1])
        top_image, top_indices = torch.topk(image_scores, top_count, dim=1)
        top_proprio = proprio_query.unsqueeze(1) @ values["proprio_feature"][top_indices].transpose(1, 2)
        total = self.image_weight * top_image + self.proprio_weight * top_proprio.squeeze(1)
        nearest = total.max(dim=1).values
        finite = nearest[torch.isfinite(nearest)]
        if len(finite) == 0:
            return (-1.0, 1.0)
        return (
            float(torch.quantile(finite, 1.0 / 3.0).item()),
            float(torch.quantile(finite, 2.0 / 3.0).item()),
        )

    @torch.no_grad()
    def _encode_dino_query(self, obs) -> torch.Tensor:
        image = obs["head_cam"][:, self.policy.n_obs_steps - 1]
        return _dinov3_features(self.dinov3, image, self._torch_device).to(self._torch_device)

    @torch.no_grad()
    def _encode_proprio_query(self, obs) -> torch.Tensor:
        normalized = self.policy.normalizer.normalize(obs)["agent_pos"]
        query = normalized[:, : self.policy.n_obs_steps].reshape(normalized.shape[0], -1)
        return F.normalize(query.to(self._torch_device).float(), dim=-1)

    @torch.no_grad()
    def _retrieve_multimodal(self, condition, image_query, proprio_query):
        values = self.memory["__global__"]
        sources, expert_conditions, metadata = [], [], []
        query_episodes = self._episode_counter * self.num_envs + np.arange(condition.shape[0])
        for env_id in range(condition.shape[0]):
            valid = torch.ones(len(values["source"]), dtype=torch.bool, device=self._torch_device)
            if self.exclude_same_episode:
                valid &= values["episode_index"].to(self._torch_device) != int(query_episodes[env_id])
            valid_indices = torch.where(valid)[0]
            if len(valid_indices) == 0:
                valid_indices = torch.arange(len(values["source"]), device=self._torch_device)
            image_scores = values["dinov3_feature"][valid_indices] @ image_query[env_id]
            top_count = min(self.retrieval_top_m, len(valid_indices))
            image_top, order = torch.topk(image_scores, top_count)
            candidates = valid_indices[order]
            proprio_scores = values["proprio_feature"][candidates] @ proprio_query[env_id]
            total_scores = self.image_weight * image_top + self.proprio_weight * proprio_scores
            best = int(torch.argmax(total_scores).item())
            index = int(candidates[best].item())
            sources.append(values["source"][index : index + 1])
            expert_conditions.append(values["condition"][index : index + 1])
            metadata.append({
                "index": index,
                "episode": int(values["episode_index"][index].item()),
                "start": int(values["global_start"][index].item()),
                "image_similarity": float(image_top[best].item()),
                "proprio_similarity": float(proprio_scores[best].item()),
                "total_similarity": float(total_scores[best].item()),
                "top_m_episode_indices": [int(values["episode_index"][item].item()) for item in candidates.detach().cpu()],
                "top_m_global_starts": [int(values["global_start"][item].item()) for item in candidates.detach().cpu()],
            })
        return torch.cat(sources, dim=0), torch.cat(expert_conditions, dim=0), metadata

    @torch.no_grad()
    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()
        policy = self.policy
        condition, visual_phase = self._encode_condition_and_visual_phase(obs)
        oracle_phase = torch.full_like(visual_phase, self._oracle_time_phase())
        image_query = self._encode_dino_query(obs)
        proprio_query = self._encode_proprio_query(obs)
        sources, expert_conditions, metadata = self._retrieve_multimodal(
            condition, image_query, proprio_query
        )
        low_sim, high_sim = self._multimodal_depth_thresholds
        generated, depth_rows = [], []
        for env_id, item in enumerate(metadata):
            similarity = item["total_similarity"]
            if similarity >= high_sim:
                target_depth, bucket = 0.8, "high_similarity_t0.8"
            elif similarity >= low_sim:
                target_depth, bucket = 0.6, "mid_similarity_t0.6"
            else:
                target_depth, bucket = 0.4, "low_similarity_t0.4"
            source = sources[env_id : env_id + 1]
            reverse_ms = 0.0
            if target_depth < 0.8:
                tic = time.perf_counter()
                source = phase_bank.midpoint_integrate(
                    policy.model, source, expert_conditions[env_id : env_id + 1],
                    0.8, target_depth, self.flow_steps
                )
                phase_bank.sync(self._torch_device)
                reverse_ms = (time.perf_counter() - tic) * 1000.0
            tic = time.perf_counter()
            generated.append(phase_bank.midpoint_integrate(
                policy.model, source, condition[env_id : env_id + 1],
                target_depth, 1.0, self.flow_steps
            ))
            phase_bank.sync(self._torch_device)
            forward_ms = (time.perf_counter() - tic) * 1000.0
            depth_rows.append((target_depth, bucket, reverse_ms, forward_ms))
        normalized_trajectory = torch.cat(generated, dim=0)
        action = policy.normalizer["action"].unnormalize(normalized_trajectory)
        normalized_for_log = policy.normalizer["action"].normalize(action)
        for env_id, item in enumerate(metadata):
            target_depth, bucket, reverse_ms, forward_ms = depth_rows[env_id]
            row = {
                "episode_id": self._episode_counter * self.num_envs + env_id,
                "query_id": self._query_counter,
                "env_id": env_id,
                "env_step": self.step,
                "method": self.method,
                "selected_phase": PHASES[int(oracle_phase[env_id].item())],
                "retrieval_top_m": self.retrieval_top_m,
                "retrieved_episode_index": item["episode"],
                "retrieved_global_start": item["start"],
                "retrieved_top_m_episode_indices": item["top_m_episode_indices"],
                "retrieved_top_m_global_starts": item["top_m_global_starts"],
                "image_similarity": item["image_similarity"],
                "proprio_similarity": item["proprio_similarity"],
                "total_similarity": item["total_similarity"],
                "low_similarity_threshold": low_sim,
                "high_similarity_threshold": high_sim,
                "adaptive_t_star": target_depth,
                "adaptive_depth_bucket": bucket,
                "reverse_ms": reverse_ms,
                "forward_ms": forward_ms,
                "source_norm": float(sources[env_id].flatten().norm()),
                "action_norm": float(normalized_for_log[env_id].flatten().norm()),
            }
            self._append_trace(row)
            self._append_retrieval_log(row)
        self._query_counter += 1
        future_start = policy.n_obs_steps - 1
        future_end = future_start + policy.n_action_steps
        return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)


class QKVActionMemoryRunner(MultimodalAdaptiveDepthRunner):
    """Action-level memory attention: observation Q/K and inverse-state V.

    This is the small action analogue of a memory-bank cross-attention block.
    The frozen Flow remains unchanged.  A current DINOv3/proprio observation
    is Q, expert observation descriptors are K, and expert x0.8 inverse states
    are V.  Attention is restricted to the top-M observation matches to keep
    the memory both cheap and interpretable.
    """

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        super()._init_policy(default_runner, **kwargs)
        config = RUNNER_CONFIG
        if config is None:
            raise RuntimeError("RUNNER_CONFIG is not initialized")
        self.qkv_top_m = max(1, int(config.get("qkv_top_m", 32)))
        self.qkv_temperature = float(config.get("qkv_temperature", 0.07))
        self.qkv_attention = ActionMemoryQKV(
            image_weight=self.image_weight,
            proprio_weight=self.proprio_weight,
            temperature=self.qkv_temperature,
            top_m=self.qkv_top_m,
        )
        self.qkv_adaptive_depth = self.method == "qkv_adaptive_depth_inverse"

    @torch.no_grad()
    def _retrieve_qkv(self, image_query: torch.Tensor, proprio_query: torch.Tensor):
        values = self.memory["__global__"]
        sources, expert_conditions, metadata = [], [], []
        query_episodes = self._episode_counter * self.num_envs + np.arange(image_query.shape[0])
        bank_episode_ids = values["episode_index"].to(self._torch_device)
        for env_id in range(image_query.shape[0]):
            valid = torch.ones(
                len(values["source"]), dtype=torch.bool, device=self._torch_device
            )
            if self.exclude_same_episode:
                valid &= bank_episode_ids != int(query_episodes[env_id])
            result = self.qkv_attention(
                query_image=image_query[env_id],
                query_proprio=proprio_query[env_id],
                key_image=values["dinov3_feature"],
                key_proprio=values["proprio_feature"],
                value_source=values["source"],
                valid_mask=valid,
            )
            sources.append(result.source)
            expert_conditions.append(values["condition"][result.top_index : result.top_index + 1])
            top_indices = result.top_indices.detach().cpu().tolist()
            metadata.append({
                "top_index": result.top_index,
                "episode": int(values["episode_index"][result.top_index].item()),
                "start": int(values["global_start"][result.top_index].item()),
                "top_indices": [int(index) for index in top_indices],
                "top_episode_indices": [
                    int(values["episode_index"][index].item()) for index in top_indices
                ],
                "top_global_starts": [
                    int(values["global_start"][index].item()) for index in top_indices
                ],
                "top_scores": [float(score) for score in result.top_scores.detach().cpu().tolist()],
                "weights": [float(weight) for weight in result.weights.detach().cpu().tolist()],
                "top1_similarity": float(result.top_scores[0].item()),
                "attention_entropy": result.entropy,
                "effective_k": result.effective_k,
            })
        return torch.stack(sources).to(self._torch_device), torch.cat(expert_conditions, dim=0), metadata

    @torch.no_grad()
    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()
        policy = self.policy
        condition, visual_phase = self._encode_condition_and_visual_phase(obs)
        oracle_phase = torch.full_like(visual_phase, self._oracle_time_phase())
        image_query = self._encode_dino_query(obs)
        proprio_query = self._encode_proprio_query(obs)
        sources, expert_conditions, metadata = self._retrieve_qkv(image_query, proprio_query)

        low_sim, high_sim = self._multimodal_depth_thresholds
        generated, depth_rows = [], []
        for env_id, item in enumerate(metadata):
            if self.qkv_adaptive_depth:
                if item["top1_similarity"] >= high_sim:
                    target_depth, bucket = 0.8, "high_similarity_t0.8"
                elif item["top1_similarity"] >= low_sim:
                    target_depth, bucket = 0.6, "mid_similarity_t0.6"
                else:
                    target_depth, bucket = 0.4, "low_similarity_t0.4"
            else:
                target_depth, bucket = 0.8, "fixed_t0.8"

            source = sources[env_id : env_id + 1]
            reverse_ms = 0.0
            if target_depth < 0.8:
                # A fused V has no single expert condition.  Use the top-1
                # item's condition only for the optional adaptive-depth
                # diagnostic; the primary QKV method has no reverse pass.
                tic = time.perf_counter()
                source = phase_bank.midpoint_integrate(
                    policy.model,
                    source,
                    expert_conditions[env_id : env_id + 1],
                    0.8,
                    target_depth,
                    self.flow_steps,
                )
                phase_bank.sync(self._torch_device)
                reverse_ms = (time.perf_counter() - tic) * 1000.0
            tic = time.perf_counter()
            generated.append(
                phase_bank.midpoint_integrate(
                    policy.model,
                    source,
                    condition[env_id : env_id + 1],
                    target_depth,
                    1.0,
                    self.flow_steps,
                )
            )
            phase_bank.sync(self._torch_device)
            forward_ms = (time.perf_counter() - tic) * 1000.0
            depth_rows.append((target_depth, bucket, reverse_ms, forward_ms))

        normalized_trajectory = torch.cat(generated, dim=0)
        action = policy.normalizer["action"].unnormalize(normalized_trajectory)
        normalized_for_log = policy.normalizer["action"].normalize(action)
        for env_id, item in enumerate(metadata):
            target_depth, bucket, reverse_ms, forward_ms = depth_rows[env_id]
            row = {
                "episode_id": self._episode_counter * self.num_envs + env_id,
                "query_id": self._query_counter,
                "env_id": env_id,
                "env_step": self.step,
                "method": self.method,
                "selected_phase": PHASES[int(oracle_phase[env_id].item())],
                "retrieval_top_m": self.qkv_top_m,
                "retrieved_top_m_episode_indices": item["top_episode_indices"],
                "retrieved_top_m_global_starts": item["top_global_starts"],
                "retrieved_topk_weights": item["weights"],
                "retrieved_episode_index": item["episode"],
                "retrieved_global_start": item["start"],
                "retrieval_similarity": item["top1_similarity"],
                "retrieval_distance": 1.0 - item["top1_similarity"],
                "qkv_attention_entropy": item["attention_entropy"],
                "qkv_effective_k": item["effective_k"],
                "qkv_temperature": self.qkv_temperature,
                "adaptive_t_star": target_depth,
                "adaptive_depth_bucket": bucket,
                "reverse_ms": reverse_ms,
                "forward_ms": forward_ms,
                "source_norm": float(sources[env_id].flatten().norm()),
                "action_norm": float(normalized_for_log[env_id].flatten().norm()),
            }
            self._append_trace(row)
            self._append_retrieval_log(row)
        self._query_counter += 1
        future_start = policy.n_obs_steps - 1
        future_end = future_start + policy.n_action_steps
        return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)


class FlowConditionQKVRunner(PhaseFilteredLocalRetrievalRunner):
    """QKV memory read using the frozen Flow condition embedding for Q/K.

    Unlike :class:`QKVActionMemoryRunner`, this path does not run DINO at
    inference.  ``condition`` is the same representation passed as
    ``global_cond`` to the frozen Flow, the memory ``condition`` tensors are
    K, and stored x0.8 inverse states are V.
    """

    def _init_policy(self, default_runner: DefaultRunner, **kwargs):
        super()._init_policy(default_runner, **kwargs)
        config = RUNNER_CONFIG
        if config is None:
            raise RuntimeError("RUNNER_CONFIG is not initialized")
        if not 0.0 <= float(self.bank_depth) <= 1.0:
            raise ValueError(f"Flow-condition QKV source depth must be in [0, 1], got {self.bank_depth}")
        self.flow_qkv_top_m = max(1, int(config.get("qkv_top_m", 32)))
        self.flow_qkv_temperature = float(config.get("qkv_temperature", 0.07))
        self.flow_qkv_value_mode = str(config.get("qkv_value_mode", "weighted"))
        if self.flow_qkv_value_mode not in {
            "weighted", "top1", "compatibility", "compatibility_prefix",
            "compatibility_release", "transition_verified",
            "temporal_compatibility", "geometric_temporal_compatibility",
        }:
            raise ValueError(f"Unknown qkv_value_mode={self.flow_qkv_value_mode!r}")
        self.flow_qkv_sequence_lock = bool(config.get("qkv_sequence_lock", False))
        # In validated-escape mode a previous expert path contributes a
        # checked continuation candidate, but never blocks global retrieval.
        self.flow_qkv_validated_escape_lock = bool(
            config.get("qkv_validated_escape_lock", False)
        )
        # Keep Flow generation unchanged, but rank real source candidates by
        # the action slice that DefaultEvalRunner actually executes.
        self.flow_qkv_executed_window_compatibility = bool(
            config.get("qkv_executed_window_compatibility", False)
        )
        # A validated successor is prepended at candidate rank 0.  When this
        # switch is enabled, validation is a real continuation decision rather
        # than merely an invitation for the local compatibility score to
        # reconsider the successor alongside unrelated trajectories.
        self.flow_qkv_force_validated_successor = bool(
            config.get("qkv_force_validated_successor", False)
        )
        if self.flow_qkv_force_validated_successor and not self.flow_qkv_validated_escape_lock:
            raise ValueError(
                "qkv_force_validated_successor requires qkv_validated_escape_lock"
            )
        self.flow_qkv_escape_semantic_tolerance = max(
            0.0, float(config.get("qkv_escape_semantic_tolerance", 0.01))
        )
        self.flow_qkv_lock_window = max(1, int(config.get("qkv_lock_window", 4)))
        configured_gate = config.get("qkv_min_similarity")
        self.flow_qkv_min_similarity = (
            None if configured_gate is None else float(configured_gate)
        )
        self.flow_qkv_attention = FlowConditionActionQKV(
            temperature=self.flow_qkv_temperature,
            top_m=self.flow_qkv_top_m,
        )
        self.flow_qkv_adaptive_depth = self.method == "flow_condition_qkv_adaptive_depth_inverse"
        self.flow_qkv_transition_verified = (
            self.flow_qkv_value_mode == "transition_verified"
            or self.flow_qkv_validated_escape_lock
        )
        # A source memory should describe a *trajectory segment*, not a bag
        # of independently retrievable actions.  In this mode the recent
        # rollout conditions/proprioceptions are matched against consecutive
        # nodes from one stored trajectory before action compatibility chooses
        # between the resulting real (never averaged) inverse states.
        self.flow_qkv_temporal = (
            self.flow_qkv_value_mode in {
                "temporal_compatibility", "geometric_temporal_compatibility",
            }
        )
        self.flow_qkv_geometric_temporal = (
            self.flow_qkv_value_mode == "geometric_temporal_compatibility"
        )
        self.flow_qkv_temporal_history = max(
            2, int(config.get("qkv_temporal_history", 3))
        )
        self.flow_qkv_spatial_pool = max(
            4, int(config.get("qkv_spatial_pool", 128))
        )
        self._recent_condition: list[deque[torch.Tensor]] = [
            deque(maxlen=self.flow_qkv_temporal_history) for _ in range(self.num_envs)
        ]
        self._recent_proprio: list[deque[torch.Tensor]] = [
            deque(maxlen=self.flow_qkv_temporal_history) for _ in range(self.num_envs)
        ]
        self._temporal_predecessor_by_index: dict[int, int] = {}
        if self.flow_qkv_temporal:
            for values in self.memory.values():
                required_spatial_key = (
                    "proprio_state" if self.flow_qkv_geometric_temporal
                    else "proprio_feature"
                )
                if required_spatial_key not in values:
                    raise RuntimeError(
                        f"{self.flow_qkv_value_mode} requires a memory with actual "
                        f"{required_spatial_key} values"
                    )
                if "proprio_feature" in values:
                    values["proprio_feature"] = F.normalize(
                        values["proprio_feature"].to(self._torch_device), dim=-1
                    )
                if "proprio_state" in values:
                    values["proprio_state"] = values["proprio_state"].to(self._torch_device)
            self._build_temporal_graph()
        # ``DefaultEvalRunner`` numbers episodes locally from zero.  A
        # continual-memory driver may evaluate one task instance per process,
        # so retain its real stream index for leave-one-demo-out exclusion and
        # for unambiguous captured-record provenance.
        self.episode_id_offset = int(config.get("episode_id_offset", 0))
        # A support rollout can optionally expose the *actual discrete x0*
        # selected under each observed condition.  The driver retains records
        # only for whole-episode successes after evaluation completes.  This
        # captures a condition--source association, never a gradient/update
        # to the frozen Flow model.
        self.capture_source_records = bool(config.get("capture_source_records", False))
        capture_dir = config.get("capture_source_dir")
        self.capture_source_dir = (
            Path(capture_dir).expanduser().resolve()
            if self.capture_source_records and capture_dir else None
        )
        if self.capture_source_dir is not None:
            self.capture_source_dir.mkdir(parents=True, exist_ok=True)
        if self.capture_source_records:
            global EPISODIC_SOURCE_CAPTURE
            EPISODIC_SOURCE_CAPTURE = []
        self._transition_successor_index = [None] * self.num_envs
        self._transition_successor_by_index: dict[int, int] = {}
        self._transition_validation_threshold = None
        if self.flow_qkv_transition_verified:
            self._build_transition_graph()

    def _runtime_episode_id(self, env_id: int) -> int:
        return self.episode_id_offset + self._episode_counter * self.num_envs + env_id

    def reset(self):
        super().reset()
        # Never carry a trajectory pointer across episodes.  A pointer is
        # merely a hypothesis until the next real observation validates it.
        self._transition_successor_index = [None] * self.num_envs
        self._recent_condition = [
            deque(maxlen=self.flow_qkv_temporal_history) for _ in range(self.num_envs)
        ]
        self._recent_proprio = [
            deque(maxlen=self.flow_qkv_temporal_history) for _ in range(self.num_envs)
        ]

    @torch.no_grad()
    def _current_proprio_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return policy-normalized (but not L2-normalized) joint history.

        The per-joint limits normalizer equalizes coordinate scale.  Keeping
        its magnitude is essential: L2 cosine of a high-dimensional joint
        history is nearly saturated even for geometrically different arms.
        """
        normalized = self.policy.normalizer.normalize(obs)["agent_pos"]
        history = normalized[:, : self.policy.n_obs_steps].reshape(normalized.shape[0], -1)
        return history.to(self._torch_device).float()

    @torch.no_grad()
    def _current_proprio_feature(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Legacy cosine key retained for the original temporal control."""
        return F.normalize(self._current_proprio_state(obs), dim=-1)

    @torch.no_grad()
    def _build_temporal_graph(self) -> None:
        """Index real predecessor nodes separated by one executed chunk."""
        values = self.memory["__global__"]
        stride = max(1, int(self.policy.n_action_steps))
        episodes = [int(value) for value in values["episode_index"].tolist()]
        starts = [int(value) for value in values["global_start"].tolist()]
        by_episode_start = {
            (episode, start): index
            for index, (episode, start) in enumerate(zip(episodes, starts))
        }
        self._temporal_predecessor_by_index = {
            index: predecessor
            for index, (episode, start) in enumerate(zip(episodes, starts))
            if (predecessor := by_episode_start.get((episode, start - stride))) is not None
        }
        if not self._temporal_predecessor_by_index:
            raise RuntimeError(
                "temporal_compatibility found no contiguous +n_action_steps memory paths"
            )

    @torch.no_grad()
    def _temporal_candidate_indices(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        env_id: int,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Rank current nodes by a contiguous recent rollout-to-memory match.

        At the first query no temporal context exists, so the usual frozen
        Flow-condition retrieval is retained.  From the second query onward,
        a candidate is admissible only when it has enough *real predecessor*
        nodes from the same stored trajectory.  Spatial (proprio) and Flow
        condition cosines are averaged at every aligned time point; neither
        source state nor feature is blended.
        """
        values = self.memory["__global__"]
        current_condition = F.normalize(condition[env_id].flatten(), dim=0)
        current_proprio = proprio[env_id]
        self._recent_condition[env_id].append(current_condition.detach())
        self._recent_proprio[env_id].append(current_proprio.detach())
        history_condition = list(self._recent_condition[env_id])
        history_proprio = list(self._recent_proprio[env_id])
        history_length = len(history_condition)

        condition_keys = F.normalize(values["condition"].flatten(1), dim=-1)
        if self.flow_qkv_geometric_temporal:
            spatial_keys = values["proprio_state"]
        else:
            spatial_keys = values["proprio_feature"]
        valid_indices = torch.where(valid)[0]
        if len(valid_indices) == 0:
            valid_indices = torch.arange(len(values["source"]), device=self._torch_device)

        # Candidate nodes must have a complete contiguous predecessor path.
        if history_length >= 2:
            contiguous = []
            for index in valid_indices.detach().cpu().tolist():
                cursor = int(index)
                for _ in range(history_length - 1):
                    cursor = self._temporal_predecessor_by_index.get(cursor, -1)
                    if cursor < 0:
                        break
                else:
                    contiguous.append(int(index))
            if contiguous:
                candidate_indices = torch.tensor(
                    contiguous, device=self._torch_device, dtype=torch.long
                )
            else:
                # A missing path is an unsupported temporal query, not an
                # excuse to fabricate continuity.  Use the static retrieval
                # as an explicit cold-start fallback.
                candidate_indices = valid_indices
                history_condition = history_condition[-1:]
                history_proprio = history_proprio[-1:]
        else:
            candidate_indices = valid_indices

        node_indices = candidate_indices.clone()
        condition_scores = torch.zeros(len(candidate_indices), device=self._torch_device)
        spatial_values = torch.zeros(len(candidate_indices), device=self._torch_device)
        reversed_history = list(zip(
            reversed(history_condition), reversed(history_proprio)
        ))
        for history_index, (condition_query, proprio_query) in enumerate(reversed_history):
            condition_scores += condition_keys[node_indices] @ condition_query
            if self.flow_qkv_geometric_temporal:
                # RMS in policy-normalized joint coordinates is an actual
                # spatial discrepancy.  It retains both joint magnitude and
                # direction, unlike the saturated cosine control above.
                spatial_values += (
                    (spatial_keys[node_indices] - proprio_query)
                    .pow(2).mean(dim=-1).sqrt()
                )
            else:
                spatial_values += spatial_keys[node_indices] @ proprio_query
            # No predecessor is needed after scoring the oldest aligned
            # query.  Advancing once more would incorrectly demand a fourth
            # node for a three-node temporal match.
            if history_index + 1 < len(reversed_history):
                node_indices = torch.tensor(
                    [self._temporal_predecessor_by_index[int(index)] for index in node_indices.tolist()],
                    device=self._torch_device,
                    dtype=torch.long,
                )
        condition_scores /= float(len(history_condition))
        spatial_values /= float(len(history_condition))
        top_count = min(self.flow_qkv_top_m, len(candidate_indices))
        if self.flow_qkv_geometric_temporal:
            # Two-stage retrieval avoids an arbitrary feature-weight: the
            # frozen Flow condition first enforces semantic/visual support;
            # within that support raw joint RMS determines spatial-temporal
            # continuity.  Sources remain separate Flow candidates.
            pool_count = min(self.flow_qkv_spatial_pool, len(candidate_indices))
            _, semantic_order = torch.topk(condition_scores, k=pool_count)
            semantic_indices = candidate_indices[semantic_order]
            semantic_spatial = spatial_values[semantic_order]
            negative_distance, spatial_order = torch.topk(-semantic_spatial, k=top_count)
            top_indices = semantic_indices[spatial_order]
            top_scores = negative_distance
            current_spatial = -semantic_spatial[spatial_order]
        else:
            scores = 0.5 * (condition_scores + spatial_values)
            top_scores, order = torch.topk(scores, k=top_count)
            top_indices = candidate_indices[order]
            current_spatial = spatial_keys[top_indices] @ current_proprio
        current_condition_scores = condition_keys[top_indices] @ current_condition
        return top_indices, top_scores, {
            "temporal_history_length": int(history_length),
            "temporal_path_matched": bool(len(history_condition) == history_length),
            "temporal_current_condition_scores": current_condition_scores.detach().cpu().tolist(),
            "temporal_current_spatial_scores": current_spatial.detach().cpu().tolist(),
            "temporal_spatial_metric": (
                "negative_normalized_joint_history_rms"
                if self.flow_qkv_geometric_temporal else "proprio_cosine"
            ),
            "temporal_semantic_pool": (
                int(min(self.flow_qkv_spatial_pool, len(candidate_indices)))
                if self.flow_qkv_geometric_temporal else None
            ),
        }

    @torch.no_grad()
    def _build_transition_graph(self) -> None:
        """Build a one-executed-chunk successor graph over real memory items.

        A node is an expert action window.  Its successor is the window from
        the same demonstration whose start advances by exactly
        ``n_action_steps``.  The acceptance threshold is calibrated only from
        expert consecutive-condition similarities; no rollout success or task
        labels are used.
        """
        values = self.memory["__global__"]
        stride = max(1, int(self.policy.n_action_steps))
        episodes = [int(value) for value in values["episode_index"].tolist()]
        starts = [int(value) for value in values["global_start"].tolist()]
        by_episode_start = {
            (episode, start): index
            for index, (episode, start) in enumerate(zip(episodes, starts))
        }
        self._transition_successor_by_index = {
            index: successor
            for index, (episode, start) in enumerate(zip(episodes, starts))
            if (successor := by_episode_start.get((episode, start + stride))) is not None
        }
        if not self._transition_successor_by_index:
            raise RuntimeError("Transition-verified retrieval found no +n_action_steps successors")
        predecessor = torch.tensor(
            list(self._transition_successor_by_index), dtype=torch.long, device=self._torch_device
        )
        successor = torch.tensor(
            [self._transition_successor_by_index[int(index)] for index in predecessor.tolist()],
            dtype=torch.long,
            device=self._torch_device,
        )
        condition = F.normalize(values["condition"].flatten(1), dim=-1)
        transition_similarity = (condition[predecessor] * condition[successor]).sum(dim=-1)
        # A 5th percentile support boundary is deliberately conservative: it
        # answers only whether the observed successor could plausibly be the
        # next local state of this expert memory path.
        self._transition_validation_threshold = float(
            torch.quantile(transition_similarity, 0.05).item()
        )

    @torch.no_grad()
    def _retrieve_flow_qkv(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ):
        values = self.memory["__global__"]
        keys = values["condition"].flatten(1)
        sources, expert_conditions, metadata = [], [], []
        query_episodes = self.episode_id_offset + self._episode_counter * self.num_envs + np.arange(condition.shape[0])
        bank_episode_ids = values["episode_index"].to(self._torch_device)
        for env_id in range(condition.shape[0]):
            valid = torch.ones(
                len(values["source"]), dtype=torch.bool, device=self._torch_device
            )
            if self.exclude_same_episode:
                valid &= bank_episode_ids != int(query_episodes[env_id])
            if self.flow_qkv_sequence_lock and not self.flow_qkv_validated_escape_lock:
                locked_episode = self._locked_episode[env_id]
                locked_start = self._locked_start[env_id]
                if locked_episode is not None and locked_start is not None:
                    same_locked_episode = valid & (bank_episode_ids == locked_episode)
                    # One rollout update executes n_action_steps actions.  A
                    # locked memory trajectory must therefore advance by at
                    # least one complete action window; using only
                    # global_start > locked_start repeatedly selected starts
                    # one frame apart and replayed almost identical chunks.
                    stride = max(1, int(self.policy.n_action_steps))
                    next_start = locked_start + stride
                    forward_only = same_locked_episode & (
                        values["global_start"].to(self._torch_device) >= next_start
                    )
                    if bool(forward_only.any()):
                        local_forward = forward_only & (
                            values["global_start"].to(self._torch_device)
                            <= locked_start + self.flow_qkv_lock_window * stride
                        )
                        valid = local_forward if bool(local_forward.any()) else forward_only
            temporal_fields: dict[str, Any] = {}
            if self.flow_qkv_temporal:
                if proprio is None:
                    raise RuntimeError("temporal_compatibility requires a proprio query")
                top_indices_tensor, top_scores_tensor, temporal_fields = self._temporal_candidate_indices(
                    condition, proprio, env_id, valid
                )
                top_index = int(top_indices_tensor[0].item())
                # All candidates remain individual, on-trajectory source
                # states.  The subsequent compatibility stage is the same
                # action-space feasibility check used by the best existing
                # discrete retrieval baseline.
                sources.append(values["source"][top_indices_tensor])
                attention_entropy = float("nan")
                effective_k = float(len(top_indices_tensor))
                weights = [1.0 / float(len(top_indices_tensor))] * len(top_indices_tensor)
            else:
                result = self.flow_qkv_attention(
                    query=condition[env_id].flatten(),
                    keys=keys,
                    value_source=values["source"],
                    valid_mask=valid,
                )
                top_indices_tensor = result.top_indices
                top_scores_tensor = result.top_scores
                top_index = int(result.top_index)
                if self.flow_qkv_value_mode == "top1":
                    sources.append(values["source"][top_index])
                elif self.flow_qkv_value_mode in {
                    "compatibility", "compatibility_prefix", "compatibility_release",
                    "transition_verified",
                }:
                    # Preserve each real inverse state as an independent action
                    # proposal.  In particular, never average sources before the
                    # Flow forward pass: latent averaging was empirically brittle.
                    sources.append(values["source"][top_indices_tensor])
                else:
                    sources.append(result.source)
                attention_entropy = result.entropy
                effective_k = result.effective_k
                weights = [float(weight) for weight in result.weights.detach().cpu().tolist()]
            expert_conditions.append(values["condition"][top_index : top_index + 1])
            # Do not advance the lock here.  In compatibility mode, the
            # QKV-top-1 proposal is not necessarily the real inverse state
            # that is propagated and executed below.  The lock is updated
            # only after action-space selection has identified that executed
            # discrete source.
            top_indices = top_indices_tensor.detach().cpu().tolist()
            metadata.append({
                "top_index": top_index,
                "top_indices": [int(index) for index in top_indices],
                "episode": int(values["episode_index"][top_index].item()),
                "start": int(values["global_start"][top_index].item()),
                "top_episode_indices": [
                    int(values["episode_index"][index].item()) for index in top_indices
                ],
                "top_global_starts": [
                    int(values["global_start"][index].item()) for index in top_indices
                ],
                "top_scores": [float(score) for score in top_scores_tensor.detach().cpu().tolist()],
                "weights": weights,
                "top1_similarity": float(top_scores_tensor[0].item()),
                "attention_entropy": attention_entropy,
                "effective_k": effective_k,
                "value_mode": self.flow_qkv_value_mode,
                "sequence_lock": self.flow_qkv_sequence_lock,
                "validated_escape_lock": self.flow_qkv_validated_escape_lock,
                **temporal_fields,
            })
        return torch.stack(sources).to(self._torch_device), torch.cat(expert_conditions, dim=0), metadata

    @torch.no_grad()
    def _transition_validate_and_augment(
        self,
        condition: torch.Tensor,
        sources: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Offer a validated local successor alongside fresh global matches.

        This is explicitly not the earlier blind sequence lock.  The previous
        selected memory node first proposes its known successor.  In
        validated-escape mode that successor is retained when its *current*
        semantic match is within a small cosine tolerance of the best fresh
        global candidate; otherwise global retrieval is allowed to release
        the stale trajectory.  Global retrieval remains available every chunk.
        """
        if not self.flow_qkv_transition_verified:
            return sources, metadata
        values = self.memory["__global__"]
        keys = F.normalize(values["condition"].flatten(1), dim=-1)
        query = F.normalize(condition.flatten(1), dim=-1)
        adjusted = sources.clone()
        for env_id, item in enumerate(metadata):
            expected_index = self._transition_successor_index[env_id]
            transition_similarity = None
            transition_reference_similarity = None
            transition_relative_gap = None
            accepted = False
            candidate_indices = list(item["top_indices"])
            candidate_scores = list(item["top_scores"])
            if expected_index is not None:
                transition_similarity = float(
                    (query[env_id] * keys[expected_index]).sum().item()
                )
                if self.flow_qkv_validated_escape_lock:
                    # An absolute expert-to-expert threshold was too strict
                    # under normal closed-loop state evolution.  The relevant
                    # question is instead whether the old path remains as
                    # semantically plausible as the freshly retrieved paths.
                    transition_reference_similarity = max(
                        item.get("temporal_current_condition_scores", [
                            float("-inf")
                        ])
                    )
                    transition_relative_gap = (
                        transition_reference_similarity - transition_similarity
                    )
                    accepted = transition_relative_gap <= self.flow_qkv_escape_semantic_tolerance
                else:
                    accepted = transition_similarity >= float(self._transition_validation_threshold)
                if accepted:
                    # Keep real source states discrete.  The validated local
                    # successor displaces only the weakest global proposal;
                    # all other candidates are still re-retrieved globally.
                    candidate_indices = [expected_index] + [
                        index for index in candidate_indices if index != expected_index
                    ]
                    candidate_indices = candidate_indices[: self.flow_qkv_top_m]
                    score_by_index = {
                        index: score
                        for index, score in zip(item["top_indices"], item["top_scores"])
                    }
                    score_by_index[expected_index] = transition_similarity
                    candidate_scores = [score_by_index[index] for index in candidate_indices]
                    adjusted[env_id] = values["source"][candidate_indices]
            item["transition_expected_index"] = expected_index
            item["transition_similarity"] = transition_similarity
            item["transition_reference_similarity"] = transition_reference_similarity
            item["transition_relative_gap"] = transition_relative_gap
            item["transition_validation_threshold"] = self._transition_validation_threshold
            item["transition_escape_semantic_tolerance"] = (
                self.flow_qkv_escape_semantic_tolerance
                if self.flow_qkv_validated_escape_lock else None
            )
            item["transition_accepted"] = accepted
            item["top_indices"] = candidate_indices
            item["top_scores"] = candidate_scores
            item["top_episode_indices"] = [
                int(values["episode_index"][index].item()) for index in candidate_indices
            ]
            item["top_global_starts"] = [
                int(values["global_start"][index].item()) for index in candidate_indices
            ]
        return adjusted, metadata

    @torch.no_grad()
    def _predict_compatibility_action(
        self,
        obs: dict[str, torch.Tensor],
        policy,
        condition: torch.Tensor,
        sources: torch.Tensor,
        metadata: list[dict[str, Any]],
        oracle_phase: torch.Tensor,
    ) -> torch.Tensor:
        """Select a discrete inverse source after reconditioning in action space.

        Observation cosine retrieval has many near ties.  For the top-M real
        x0 states we therefore generate M actions under the *current*
        condition and select the source that is both locally executable and
        least rewritten relative to its expert action provenance.  This is a
        parameter-free source-compatibility test, not a latent interpolation
        or a learned critic.
        """
        if self.flow_qkv_adaptive_depth:
            raise ValueError("compatibility selection is defined for a fixed source depth")
        if self.flow_qkv_min_similarity is not None:
            raise ValueError("Do not combine compatibility selection with a similarity gate")

        batch_size, candidate_count = sources.shape[:2]
        flat_sources = sources.reshape(batch_size * candidate_count, *sources.shape[2:])
        flat_condition = condition.repeat_interleave(candidate_count, dim=0)
        tic = time.perf_counter()
        normalized_candidates = phase_bank.midpoint_integrate(
            policy.model,
            flat_sources,
            flat_condition,
            float(self.bank_depth),
            1.0,
            self.flow_steps,
        ).reshape(batch_size, candidate_count, *sources.shape[2:])
        phase_bank.sync(self._torch_device)
        forward_ms = (time.perf_counter() - tic) * 1000.0
        action_candidates = policy.normalizer["action"].unnormalize(normalized_candidates)

        values = self.memory["__global__"]
        candidate_indices = torch.tensor(
            [item["top_indices"] for item in metadata],
            dtype=torch.long,
            device=self._torch_device,
        )
        expert_actions = values["raw_action"][candidate_indices]
        future_start = policy.n_obs_steps - 1
        future_end = future_start + policy.n_action_steps
        captured_proprio_state = (
            self._current_proprio_state(obs) if self.capture_source_records else None
        )
        captured_proprio = (
            F.normalize(captured_proprio_state, dim=-1)
            if captured_proprio_state is not None else None
        )
        current_joint = obs["agent_pos"][:, future_start].to(
            device=self._torch_device, dtype=action_candidates.dtype
        )

        # The first executed position target must connect to the observed arm
        # state.  The provenance term picks the source whose behavior remains
        # most compatible after condition rewrite.  Both are RMS distances in
        # action units, so their unweighted sum has no tuned coefficient.
        boundary_error = (
            (action_candidates[:, :, future_start] - current_joint[:, None])
            .flatten(2)
            .norm(dim=-1)
            / math.sqrt(float(current_joint.shape[-1]))
        )
        # Default compatibility measures full-horizon transport.  The prefix
        # variant uses exactly the [7:15] chunk the runner will execute.  It
        # is not a coefficient sweep: it removes an otherwise irrelevant
        # unexecuted tail from source selection.
        rewrite_start, rewrite_end = (future_start, future_end) if (
            self.flow_qkv_value_mode == "compatibility_prefix"
            or self.flow_qkv_executed_window_compatibility
        ) else (0, action_candidates.shape[2])
        rewrite_error = (
            (action_candidates[:, :, rewrite_start:rewrite_end]
             - expert_actions[:, :, rewrite_start:rewrite_end])
            .flatten(2)
            .norm(dim=-1)
            / math.sqrt(float(
                action_candidates.shape[-1] * (rewrite_end - rewrite_start)
            ))
        )
        compatibility_cost = boundary_error + rewrite_error
        selected = torch.argmin(compatibility_cost, dim=1)
        forced_successor = torch.zeros(
            batch_size, dtype=torch.bool, device=self._torch_device
        )
        if self.flow_qkv_force_validated_successor:
            for env_id, item in enumerate(metadata):
                if not bool(item.get("transition_accepted", False)):
                    continue
                expected_index = item.get("transition_expected_index")
                if expected_index is None or item["top_indices"][0] != expected_index:
                    raise RuntimeError(
                        "Validated successor must be candidate rank 0 before forced selection"
                    )
                selected[env_id] = 0
                forced_successor[env_id] = True
        batch_index = torch.arange(batch_size, device=self._torch_device)
        normalized_trajectory = normalized_candidates[batch_index, selected]
        action = action_candidates[batch_index, selected]

        # A cosine threshold was not a reliable trust signal: successful and
        # failed rollouts had nearly identical retrieval similarities.  In
        # ``compatibility_release`` mode we instead compare the best discrete
        # memory proposal against one native Gaussian source *after both have
        # been propagated by the same frozen Flow solver*.  There is no latent
        # blending and no learned gate.
        release_mode = self.flow_qkv_value_mode == "compatibility_release"
        gaussian_boundary_error = None
        gaussian_smoothness_error = None
        memory_feasibility_cost = None
        gaussian_feasibility_cost = None
        source_released = torch.zeros(batch_size, dtype=torch.bool, device=self._torch_device)
        if release_mode:
            memory_executed = action[:, future_start:future_end]
            memory_smoothness_error = (
                (memory_executed[:, 1:] - memory_executed[:, :-1])
                .flatten(1)
                .norm(dim=-1)
                / math.sqrt(float(max(1, (future_end - future_start - 1) * current_joint.shape[-1])))
            )
            memory_feasibility_cost = boundary_error[batch_index, selected] + memory_smoothness_error

            gaussian_source = torch.randn_like(sources[batch_index, selected])
            tic = time.perf_counter()
            gaussian_normalized = phase_bank.midpoint_integrate(
                policy.model,
                gaussian_source,
                condition,
                float(self.bank_depth),
                1.0,
                self.flow_steps,
            )
            phase_bank.sync(self._torch_device)
            forward_ms += (time.perf_counter() - tic) * 1000.0
            gaussian_action = policy.normalizer["action"].unnormalize(gaussian_normalized)
            gaussian_boundary_error = (
                (gaussian_action[:, future_start] - current_joint)
                .flatten(1)
                .norm(dim=-1)
                / math.sqrt(float(current_joint.shape[-1]))
            )
            gaussian_executed = gaussian_action[:, future_start:future_end]
            gaussian_smoothness_error = (
                (gaussian_executed[:, 1:] - gaussian_executed[:, :-1])
                .flatten(1)
                .norm(dim=-1)
                / math.sqrt(float(max(1, (future_end - future_start - 1) * current_joint.shape[-1])))
            )
            gaussian_feasibility_cost = gaussian_boundary_error + gaussian_smoothness_error
            source_released = gaussian_feasibility_cost < memory_feasibility_cost
            normalized_trajectory = torch.where(
                source_released[:, None, None], gaussian_normalized, normalized_trajectory
            )
            action = torch.where(source_released[:, None, None], gaussian_action, action)

        for env_id, item in enumerate(metadata):
            choice = int(selected[env_id].item())
            top_indices = item["top_indices"]
            chosen_memory_index = top_indices[choice]
            if self.flow_qkv_sequence_lock and not (
                release_mode and bool(source_released[env_id].item())
            ):
                # Temporal continuity must follow the actual source that
                # generated the executed chunk, not merely the QKV ranking.
                self._locked_episode[env_id] = int(
                    values["episode_index"][chosen_memory_index].item()
                )
                self._locked_start[env_id] = int(
                    values["global_start"][chosen_memory_index].item()
                )
            if self.capture_source_records:
                chosen_source = (
                    gaussian_source[env_id]
                    if release_mode and bool(source_released[env_id].item())
                    else sources[env_id, choice]
                )
                # Keep all tensors in the exact normalized representation
                # expected by the existing retrieval memory.  The episode
                # outcome is intentionally unknown here; the driver filters
                # records only after DefaultEvalRunner writes SuccessOnce.
                assert EPISODIC_SOURCE_CAPTURE is not None
                capture_record = {
                    "episode_id": self._runtime_episode_id(env_id),
                    "query_id": self._query_counter,
                    "env_step": self.step,
                    "condition": condition[env_id].detach().flatten().cpu().contiguous(),
                    "source": chosen_source.detach().cpu().contiguous(),
                    "raw_action": normalized_trajectory[env_id].detach().cpu().contiguous(),
                    "proprio_feature": captured_proprio[env_id].detach().cpu().contiguous(),
                    "proprio_state": captured_proprio_state[env_id].detach().cpu().contiguous(),
                    "chunk_stride": int(policy.n_action_steps),
                    "actual_source_used": (
                        "gaussian" if release_mode and bool(source_released[env_id].item())
                        else "memory"
                    ),
                }
                EPISODIC_SOURCE_CAPTURE.append(capture_record)
                # DefaultEvalRunner force-closes the Isaac Python process at
                # the end of evaluation.  Persist each query now so an
                # external continual controller can later filter by its
                # official SuccessOnce file without modifying that runner.
                if self.capture_source_dir is not None:
                    torch.save(
                        capture_record,
                        self.capture_source_dir / (
                            f"episode_{capture_record['episode_id']:06d}_"
                            f"query_{capture_record['query_id']:06d}.pt"
                        ),
                    )
            row = {
                "episode_id": self._runtime_episode_id(env_id),
                "query_id": self._query_counter,
                "env_id": env_id,
                "env_step": self.step,
                "method": self.method,
                "selected_phase": PHASES[int(oracle_phase[env_id].item())],
                "selected_bank": "__global__",
                "retrieval_top_m": candidate_count,
                "retrieved_top_m_episode_indices": item["top_episode_indices"],
                "retrieved_top_m_global_starts": item["top_global_starts"],
                "retrieved_topk_weights": item["weights"],
                "candidate_selected_rank": choice,
                "candidate_forced_validated_successor": bool(
                    forced_successor[env_id].item()
                ),
                "candidate_compatibility_costs": compatibility_cost[env_id].detach().cpu().tolist(),
                "candidate_boundary_errors": boundary_error[env_id].detach().cpu().tolist(),
                "candidate_rewrite_errors": rewrite_error[env_id].detach().cpu().tolist(),
                "candidate_rewrite_scope": (
                    "executed_slice" if rewrite_start == future_start
                    else "full_horizon"
                ),
                "transition_expected_index": item.get("transition_expected_index"),
                "transition_similarity": item.get("transition_similarity"),
                "transition_validation_threshold": item.get("transition_validation_threshold"),
                "transition_accepted": item.get("transition_accepted", False),
                "temporal_history_length": item.get("temporal_history_length"),
                "temporal_path_matched": item.get("temporal_path_matched"),
                "temporal_current_condition_scores": item.get("temporal_current_condition_scores"),
                "temporal_current_spatial_scores": item.get("temporal_current_spatial_scores"),
                "temporal_spatial_metric": item.get("temporal_spatial_metric"),
                "temporal_semantic_pool": item.get("temporal_semantic_pool"),
                "memory_feasibility_cost": (
                    None if memory_feasibility_cost is None
                    else float(memory_feasibility_cost[env_id].item())
                ),
                "gaussian_feasibility_cost": (
                    None if gaussian_feasibility_cost is None
                    else float(gaussian_feasibility_cost[env_id].item())
                ),
                "gaussian_boundary_error": (
                    None if gaussian_boundary_error is None
                    else float(gaussian_boundary_error[env_id].item())
                ),
                "gaussian_smoothness_error": (
                    None if gaussian_smoothness_error is None
                    else float(gaussian_smoothness_error[env_id].item())
                ),
                "actual_source_used": (
                    "gaussian" if bool(source_released[env_id].item()) else "memory"
                ),
                "retrieved_episode_index": int(values["episode_index"][chosen_memory_index].item()),
                "retrieved_global_start": int(values["global_start"][chosen_memory_index].item()),
                "retrieval_similarity": item["top_scores"][choice],
                "retrieval_distance": 1.0 - item["top_scores"][choice],
                "qkv_attention_entropy": item["attention_entropy"],
                "qkv_effective_k": item["effective_k"],
                "qkv_temperature": self.flow_qkv_temperature,
                "qkv_value_mode": self.flow_qkv_value_mode,
                "qkv_sequence_lock": self.flow_qkv_sequence_lock,
                "qkv_temporal_history": (
                    self.flow_qkv_temporal_history if self.flow_qkv_temporal else None
                ),
                "qkv_min_similarity": None,
                "qkv_gate_accept": True,
                "adaptive_t_star": float(self.bank_depth),
                "adaptive_depth_bucket": f"fixed_t{self.bank_depth:.1f}",
                "reverse_ms": 0.0,
                "forward_ms": forward_ms,
                "source_norm": float(
                    (gaussian_source[env_id] if release_mode and bool(source_released[env_id].item())
                     else sources[env_id, choice]).flatten().norm()
                ),
                "action_norm": float(normalized_trajectory[env_id].flatten().norm()),
            }
            self._append_trace(row)
            self._append_retrieval_log(row)
            if self.flow_qkv_transition_verified:
                self._transition_successor_index[env_id] = self._transition_successor_by_index.get(
                    chosen_memory_index
                )
        self._query_counter += 1
        return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)

    @torch.no_grad()
    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()
        policy = self.policy
        condition, visual_phase = self._encode_condition_and_visual_phase(obs)
        oracle_phase = torch.full_like(visual_phase, self._oracle_time_phase())
        proprio = (
            self._current_proprio_state(obs) if self.flow_qkv_geometric_temporal
            else self._current_proprio_feature(obs) if self.flow_qkv_temporal
            else None
        )
        sources, expert_conditions, metadata = self._retrieve_flow_qkv(condition, proprio)

        if self.flow_qkv_value_mode in {
            "compatibility", "compatibility_prefix", "compatibility_release",
            "transition_verified", "temporal_compatibility",
            "geometric_temporal_compatibility",
        }:
            sources, metadata = self._transition_validate_and_augment(
                condition, sources, metadata
            )
            return self._predict_compatibility_action(
                obs, policy, condition, sources, metadata, oracle_phase
            )

        # Do not force a retrieved inverse state when the query is below an
        # explicitly configured support threshold.  In that case preserve
        # the native frozen-Flow Gaussian behavior for this query.
        gate_mask = torch.zeros(
            condition.shape[0], dtype=torch.bool, device=self._torch_device
        )
        native_normalized_full = None
        if self.flow_qkv_min_similarity is not None:
            gate_mask = torch.tensor(
                [
                    item["top1_similarity"] < self.flow_qkv_min_similarity
                    for item in metadata
                ],
                dtype=torch.bool,
                device=self._torch_device,
            )
            if bool(gate_mask.any()):
                native_result = policy.predict_action(obs)
                phase_bank.sync(self._torch_device)
                native_normalized_full = policy.normalizer["action"].normalize(
                    native_result["action_pred"]
                ).to(torch.float32)

        low_sim, high_sim = self._adaptive_similarity_thresholds
        generated, depth_rows = [], []
        for env_id, item in enumerate(metadata):
            if self.flow_qkv_adaptive_depth:
                if item["top1_similarity"] >= high_sim:
                    target_depth, bucket = 0.8, "high_similarity_t0.8"
                elif item["top1_similarity"] >= low_sim:
                    target_depth, bucket = 0.6, "mid_similarity_t0.6"
                else:
                    target_depth, bucket = 0.4, "low_similarity_t0.4"
            else:
                target_depth, bucket = float(self.bank_depth), f"fixed_t{self.bank_depth:.1f}"

            source = sources[env_id : env_id + 1]
            reverse_ms = 0.0
            if target_depth < float(self.bank_depth):
                tic = time.perf_counter()
                source = phase_bank.midpoint_integrate(
                    policy.model,
                    source,
                    expert_conditions[env_id : env_id + 1],
                    float(self.bank_depth),
                    target_depth,
                    self.flow_steps,
                )
                phase_bank.sync(self._torch_device)
                reverse_ms = (time.perf_counter() - tic) * 1000.0
            tic = time.perf_counter()
            generated.append(
                phase_bank.midpoint_integrate(
                    policy.model,
                    source,
                    condition[env_id : env_id + 1],
                    target_depth,
                    1.0,
                    self.flow_steps,
                )
            )
            phase_bank.sync(self._torch_device)
            forward_ms = (time.perf_counter() - tic) * 1000.0
            depth_rows.append((target_depth, bucket, reverse_ms, forward_ms))

        normalized_trajectory = torch.cat(generated, dim=0)
        if native_normalized_full is not None:
            normalized_trajectory[gate_mask] = native_normalized_full[gate_mask]
        action = policy.normalizer["action"].unnormalize(normalized_trajectory)
        normalized_for_log = policy.normalizer["action"].normalize(action)
        for env_id, item in enumerate(metadata):
            target_depth, bucket, reverse_ms, forward_ms = depth_rows[env_id]
            row = {
                "episode_id": self._runtime_episode_id(env_id),
                "query_id": self._query_counter,
                "env_id": env_id,
                "env_step": self.step,
                "method": self.method,
                "selected_phase": PHASES[int(oracle_phase[env_id].item())],
                "selected_bank": "__global__",
                "retrieval_top_m": self.flow_qkv_top_m,
                "retrieved_top_m_episode_indices": item["top_episode_indices"],
                "retrieved_top_m_global_starts": item["top_global_starts"],
                "retrieved_topk_weights": item["weights"],
                "retrieved_episode_index": item["episode"],
                "retrieved_global_start": item["start"],
                "retrieval_similarity": item["top1_similarity"],
                "retrieval_distance": 1.0 - item["top1_similarity"],
                "qkv_attention_entropy": item["attention_entropy"],
                "qkv_effective_k": item["effective_k"],
                "qkv_temperature": self.flow_qkv_temperature,
                "qkv_value_mode": item["value_mode"],
                "qkv_sequence_lock": item["sequence_lock"],
                "qkv_min_similarity": self.flow_qkv_min_similarity,
                "qkv_gate_accept": not bool(gate_mask[env_id].item()),
                "adaptive_t_star": target_depth,
                "adaptive_depth_bucket": bucket,
                "reverse_ms": reverse_ms,
                "forward_ms": forward_ms,
                "source_norm": float(sources[env_id].flatten().norm()),
                "action_norm": float(normalized_for_log[env_id].flatten().norm()),
            }
            self._append_trace(row)
            self._append_retrieval_log(row)
            if self.flow_qkv_sequence_lock and not bool(gate_mask[env_id].item()):
                selected_memory_index = int(item["top_index"])
                self._locked_episode[env_id] = int(
                    self.memory["__global__"]["episode_index"][selected_memory_index].item()
                )
                self._locked_start[env_id] = int(
                    self.memory["__global__"]["global_start"][selected_memory_index].item()
                )
        self._query_counter += 1
        future_start = policy.n_obs_steps - 1
        future_end = future_start + policy.n_action_steps
        return action[:, future_start:future_end].transpose(0, 1).to(torch.float32)


class PhaseFilteredLocalRetrievalDefaultRunner(DefaultRunner):
    @staticmethod
    def get_eval_runner_class():
        if RUNNER_CONFIG and RUNNER_CONFIG.get("method") in {
            "multimodal_adaptive_depth_inverse",
            "qkv_inverse",
            "qkv_adaptive_depth_inverse",
        }:
            if RUNNER_CONFIG.get("method") in {"qkv_inverse", "qkv_adaptive_depth_inverse"}:
                return QKVActionMemoryRunner
            return MultimodalAdaptiveDepthRunner
        if RUNNER_CONFIG and RUNNER_CONFIG.get("method") in {
            "flow_condition_qkv_inverse",
            "flow_condition_qkv_adaptive_depth_inverse",
        }:
            return FlowConditionQKVRunner
        return PhaseFilteredLocalRetrievalRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "build_multimodal", "rollout", "summarize", "all"], default="build")
    parser.add_argument("--zarr-path", default="/data/yiming/MomentVLA-main/data_policy/pick_cubeIsaacSimL0_obs:joint_pos_act:joint_pos_100.zarr")
    parser.add_argument("--demo-root", default="/data/yiming/MomentVLA-main/roboverse_demo/demo_isaacsim/pick_cube-isaac100/robot-franka/success")
    parser.add_argument("--checkpoint", default="/data/yiming/MomentVLA-main/il_outputs/fm_unet/pick_cube_isaac100/checkpoints/30.ckpt")
    parser.add_argument("--base-bank-path", default="/data/yiming/MomentVLA-main/il_outputs/visual_phase_inversion_skill_bank_pickcube/skill_bank.pt")
    parser.add_argument("--memory-path", default="/data/yiming/MomentVLA-main/il_outputs/phase_filtered_local_inverse_retrieval_pickcube/retrieval_memory.pt")
    parser.add_argument("--multimodal-memory-path", default="/data/yiming/MomentVLA-main/il_outputs/stack_cube_n1_warmstart_bank/retrieval_memory_multimodal_08.pt")
    parser.add_argument("--dinov3-model-path", default="/data/shared/models/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-top-m", type=int, default=32)
    parser.add_argument("--qkv-top-m", type=int, default=32)
    parser.add_argument("--qkv-temperature", type=float, default=0.07)
    parser.add_argument(
        "--qkv-value-mode",
        choices=[
            "weighted", "top1", "compatibility", "compatibility_prefix",
            "compatibility_release", "transition_verified",
            "temporal_compatibility", "geometric_temporal_compatibility",
        ],
        default="weighted",
    )
    parser.add_argument("--qkv-sequence-lock", action="store_true")
    parser.add_argument(
        "--qkv-executed-window-compatibility",
        action="store_true",
        help="Rank discrete inverse sources using only the action slice executed by the runner.",
    )
    parser.add_argument(
        "--qkv-force-validated-successor",
        action="store_true",
        help=(
            "Force candidate rank 0 when it is the accepted successor of the "
            "previous expert-memory node; otherwise retain global Top-M selection."
        ),
    )
    parser.add_argument(
        "--qkv-validated-escape-lock",
        action="store_true",
        help=(
            "Keep global geometric-temporal retrieval active; admit the previous "
            "expert-path successor only after transition-support validation."
        ),
    )
    parser.add_argument(
        "--qkv-escape-semantic-tolerance",
        type=float,
        default=0.01,
        help=(
            "Validated-escape continuation is retained when its current cosine "
            "match is no worse than this amount below the best global candidate."
        ),
    )
    parser.add_argument("--qkv-lock-window", type=int, default=4)
    parser.add_argument(
        "--qkv-temporal-history",
        type=int,
        default=3,
        help="Consecutive current/memory chunk states required by temporal_compatibility.",
    )
    parser.add_argument(
        "--qkv-spatial-pool",
        type=int,
        default=128,
        help="Flow-condition Top-M pool before raw-proprio temporal reranking.",
    )
    parser.add_argument("--qkv-min-similarity", type=float, default=None)
    parser.add_argument("--image-weight", type=float, default=0.8)
    parser.add_argument("--proprio-weight", type=float, default=0.2)
    parser.add_argument("--output-dir", default="/data/yiming/MomentVLA-main/il_outputs/phase_filtered_local_inverse_retrieval_pickcube/rollouts")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--bank-episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-demos", type=int, default=50)
    parser.add_argument(
        "--task-id-start",
        type=int,
        default=0,
        help="First task/demo index to evaluate (used by continual one-episode streams).",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--task", default="pick_cube")
    parser.add_argument("--robot", default="franka")
    parser.add_argument("--sim", default="isaacsim")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth", type=float, default=0.6)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--inverse-steps", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--n-obs-steps", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--purity-threshold", type=float, default=0.80)
    parser.add_argument("--methods", default="vanilla_gaussian_flow,global_random_inverse_bank,phase_random_bank,global_nearest_inverse,phase_nearest_inverse,wrong_phase_nearest,phase_nearest_raw_replay")
    parser.add_argument("--shifts", default="0,1,2,3")
    parser.add_argument("--rollout-depths", default="0.6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-video-freq", type=int, default=1000000)
    parser.add_argument("--allow-same-episode", action="store_true")
    parser.add_argument("--retrieval-topk", type=int, default=1)
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--max-candidate-dispersion", type=float, default=0.05)
    parser.add_argument("--phase-source", choices=["oracle", "visual"], default="oracle")
    parser.add_argument(
        "--capture-source-records",
        action="store_true",
        help="Capture selected x0/condition tuples for post-rollout success-only memory insertion.",
    )
    parser.add_argument(
        "--capture-source-dir",
        default=None,
        help="Directory for per-query source records; required for cross-process continual insertion.",
    )
    parser.add_argument("--use-ema", action="store_true", default=True)
    return parser.parse_args()


def run_one_rollout(args: argparse.Namespace, method: str, shift_cm: float, depth: float) -> dict[str, Any]:
    run_dir = Path(args.output_dir).expanduser().resolve() / method / f"shift_{shift_cm:g}cm" / f"depth_{depth:.1f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "phase_trace.jsonl"
    retrieval_log_path = run_dir / "retrieval_log.jsonl"
    for path in (trace_path, retrieval_log_path):
        if path.exists():
            path.unlink()
    config = {
        "bank_path": str(Path(args.memory_path).expanduser().resolve()),
        "method": method,
        "phase_source": args.phase_source,
        "depth": depth,
        "flow_steps": args.flow_steps,
        "seed": args.seed,
        "trace_path": str(trace_path),
        "retrieval_log_path": str(retrieval_log_path),
        "run_dir": str(run_dir),
        "exclude_same_episode": not args.allow_same_episode,
        "retrieval_topk": args.retrieval_topk,
        "retrieval_top_m": getattr(args, "retrieval_top_m", 32),
        "qkv_top_m": getattr(args, "qkv_top_m", 32),
        "qkv_temperature": getattr(args, "qkv_temperature", 0.07),
        "qkv_value_mode": getattr(args, "qkv_value_mode", "weighted"),
        "qkv_sequence_lock": getattr(args, "qkv_sequence_lock", False),
        "qkv_executed_window_compatibility": getattr(
            args, "qkv_executed_window_compatibility", False
        ),
        "qkv_force_validated_successor": getattr(
            args, "qkv_force_validated_successor", False
        ),
        "qkv_validated_escape_lock": getattr(args, "qkv_validated_escape_lock", False),
        "qkv_escape_semantic_tolerance": getattr(args, "qkv_escape_semantic_tolerance", 0.01),
        "qkv_lock_window": getattr(args, "qkv_lock_window", 4),
        "qkv_temporal_history": getattr(args, "qkv_temporal_history", 3),
        "qkv_spatial_pool": getattr(args, "qkv_spatial_pool", 128),
        "qkv_min_similarity": getattr(args, "qkv_min_similarity", None),
        "dinov3_model_path": getattr(args, "dinov3_model_path", ""),
        "image_weight": getattr(args, "image_weight", 0.8),
        "proprio_weight": getattr(args, "proprio_weight", 0.2),
        "min_similarity": args.min_similarity,
        "max_candidate_dispersion": args.max_candidate_dispersion,
        "capture_source_records": bool(getattr(args, "capture_source_records", False)),
        "capture_source_dir": getattr(args, "capture_source_dir", None),
        "episode_id_offset": int(getattr(args, "task_id_start", 0)),
    }
    global RUNNER_CONFIG
    RUNNER_CONFIG = config
    phase_bank.RUNNER_CONFIG = config
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_payload = torch.load(
        checkpoint_path.open("rb"), pickle_module=__import__("dill"), map_location="cpu"
    )
    workspace = PhaseFilteredLocalRetrievalDefaultRunner(checkpoint_payload["cfg"], output_dir=str(run_dir))
    eval_args = workspace.eval_args
    eval_args.task = args.task
    eval_args.robot = args.robot
    eval_args.sim = args.sim
    eval_args.num_envs = args.num_envs
    eval_args.max_demo = args.max_demos
    eval_args.task_id_range_low = int(getattr(args, "task_id_start", 0))
    eval_args.task_id_range_high = eval_args.task_id_range_low + args.max_demos
    eval_args.max_step = args.max_steps
    eval_args.headless = True
    eval_args.gpu_id = args.gpu_id
    eval_args.level = 0
    eval_args.scene_mode = 0
    eval_args.randomization_seed = args.seed
    eval_args.cube_shift_x = float(shift_cm) / 100.0
    eval_args.cube_shift_y = 0.0
    eval_args.cube_shift_object = "cube"
    eval_args.cube_shift_step = -1
    eval_args.save_video_freq = args.save_video_freq
    eval_args.subset = f"phase_filtered_local_{method}_{shift_cm:g}cm_d{depth:.1f}"
    started = time.perf_counter()
    # A continual-memory controller may supply a reproducible trajectory file
    # with randomized initial joints.  The override is process-local and the
    # original registered task path is restored immediately; DefaultEvalRunner
    # itself is not modified.
    # Generic child-process trajectory override used by the continual-memory
    # controller.  Retain the old StackCube name only as a backward-compatible
    # fallback for archived commands; DefaultEvalRunner itself is untouched.
    trajectory_override = (
        os.environ.get("EVAL_TRAJ_PATH_OVERRIDE")
        or os.environ.get("STACKCUBE_EVAL_TRAJ_PATH")
    )
    task_cls = None
    original_trajectory_path = None
    if trajectory_override:
        from metasim.task.registry import get_task_class
        task_cls = get_task_class(args.task)
        original_trajectory_path = task_cls.traj_filepath
        task_cls.traj_filepath = trajectory_override
    try:
        workspace.evaluate(ckpt_path=checkpoint_path)
    finally:
        if task_cls is not None:
            task_cls.traj_filepath = original_trajectory_path
    result = phase_bank.parse_final_stats(run_dir)
    if bool(getattr(args, "capture_source_records", False)):
        # The runner only observes actions; DefaultEvalRunner remains the
        # authority for SuccessOnce.  Persist support tuples after filtering
        # by that final episode-level outcome, never per-chunk heuristics.
        success_map = _read_success_map(run_dir)
        captured = EPISODIC_SOURCE_CAPTURE or []
        successful_records = [
            record for record in captured
            if bool(success_map.get(int(record["episode_id"]), False))
        ]
        capture_path = run_dir / "successful_episodic_source_records.pt"
        torch.save({
            "records": successful_records,
            "all_record_count": len(captured),
            "successful_record_count": len(successful_records),
            "successful_episode_ids": sorted(
                episode for episode, success in success_map.items() if success
            ),
            "source_depth": float(depth),
            "selection": str(getattr(args, "qkv_value_mode", "weighted")),
        }, capture_path)
        result.update({
            "captured_record_path": str(capture_path),
            "captured_all_records": len(captured),
            "captured_successful_records": len(successful_records),
            "captured_successful_episodes": int(sum(success_map.values())),
        })
    result.update({"method": method, "shift_cm": shift_cm, "depth": depth, "elapsed_seconds": time.perf_counter() - started, "run_dir": str(run_dir)})
    return result


def run_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    shifts = [float(value) for value in args.shifts.split(",")]
    depths = [float(value) for value in args.rollout_depths.split(",")]
    rows = []
    for depth in depths:
        for shift in shifts:
            for method in methods:
                print(f"[rollout] method={method} shift_cm={shift:g} depth={depth:.1f}", flush=True)
                rows.append(run_one_rollout(args, method, shift, depth))
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_csv(output_dir / "rollout_results.csv", rows)
    write_csv(output_dir / "clean_rollout.csv", [row for row in rows if row["shift_cm"] == 0])
    write_csv(output_dir / "ood_rollout.csv", [row for row in rows if row["shift_cm"] != 0])
    summary = {
        "status": "completed_rollouts",
        "methods": methods,
        "shifts_cm": shifts,
        "depths": depths,
        "num_conditions": len(rows),
        "trials_per_condition": args.max_demos,
        "same_initial_states_and_seed": True,
        "exclude_same_episode": not args.allow_same_episode,
        "retrieval_topk": args.retrieval_topk,
        "min_similarity": args.min_similarity,
        "max_candidate_dispersion": args.max_candidate_dispersion,
        "default_eval_runner": True,
        "action_protocol": "DefaultEvalRunner; native Gaussian baseline or retrieved source; same action chunk",
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _read_success_map(run_dir: Path) -> dict[int, bool]:
    result = {}
    for path in run_dir.rglob("*.txt"):
        match = re.search(r"(\d+)\.txt$", path.name)
        if not match:
            continue
        success = re.search(r"SuccessOnce:\s*(True|False)", path.read_text(errors="replace"))
        if success:
            result[int(match.group(1))] = success.group(1) == "True"
    return result


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows, retrieval_rows, failure_rows = [], [], []
    for trace_path in sorted(output_dir.rglob("phase_trace.jsonl")):
        run_dir = trace_path.parent
        if len(run_dir.parts) < 3:
            continue
        method = run_dir.parts[-3]
        shift_match = re.search(r"shift_([-+0-9.eE]+)cm", run_dir.parts[-2])
        depth_match = re.search(r"depth_([-+0-9.eE]+)", run_dir.parts[-1])
        if not shift_match or not depth_match:
            continue
        shift = float(shift_match.group(1))
        depth = float(depth_match.group(1))
        rows.append({"method": method, "shift_cm": shift, "depth": depth, **phase_bank.parse_final_stats(run_dir)})
        success_map = _read_success_map(run_dir)
        traces = [json.loads(line) for line in trace_path.read_text(errors="replace").splitlines() if line.strip()]
        by_episode: dict[int, list[dict[str, Any]]] = {}
        for trace in traces:
            by_episode.setdefault(int(trace.get("episode_id", -1)), []).append(trace)
        for episode_id, episode_trace in by_episode.items():
            if episode_id < 0:
                continue
            last = episode_trace[-1]
            failure_rows.append({
                "method": method,
                "shift_cm": shift,
                "depth": depth,
                "episode_id": episode_id,
                "success": success_map.get(episode_id),
                "last_selected_phase": last.get("selected_phase"),
                "last_selected_bank": last.get("selected_bank"),
                "num_policy_queries": len(episode_trace),
            })
        retrieval_path = run_dir / "retrieval_log.jsonl"
        if retrieval_path.exists():
            retrieval_rows.extend(
                json.loads(line) for line in retrieval_path.read_text(errors="replace").splitlines() if line.strip()
            )
    rows.sort(key=lambda row: (row["shift_cm"], row["method"], row["depth"]))
    write_csv(output_dir / "rollout_results.csv", rows)
    write_csv(output_dir / "clean_rollout.csv", [row for row in rows if row["shift_cm"] == 0])
    write_csv(output_dir / "ood_rollout.csv", [row for row in rows if row["shift_cm"] != 0])
    write_csv(output_dir / "retrieval_log.csv", retrieval_rows)
    write_csv(output_dir / "phase_failure_analysis.csv", failure_rows)

    distance_rows = []
    for method in sorted({row.get("method") for row in retrieval_rows}):
        values = [row for row in retrieval_rows if row.get("method") == method and math.isfinite(float(row.get("retrieval_distance", "nan")))]
        if not values:
            continue
        distances = np.asarray([float(row["retrieval_distance"]) for row in values])
        q1, q2 = np.quantile(distances, [1 / 3, 2 / 3])
        for bucket, mask in [("near", distances <= q1), ("medium", (distances > q1) & (distances <= q2)), ("far", distances > q2)]:
            chosen = [row for row, keep in zip(values, mask) if keep]
            distance_rows.append({
                "method": method,
                "bucket": bucket,
                "num_queries": len(chosen),
                "mean_distance": float(np.mean([float(row["retrieval_distance"]) for row in chosen])) if chosen else float("nan"),
            })
    write_csv(output_dir / "retrieval_distance_analysis.csv", distance_rows)

    paired_rows = []
    shifts = sorted({row["shift_cm"] for row in rows})
    for shift in shifts:
        shift_rows = [row for row in rows if row["shift_cm"] == shift]
        maps = {row["method"]: _read_success_map(Path(row["run_dir"])) for row in shift_rows}
        for episode_id in range(args.max_demos):
            paired = {"shift_cm": shift, "episode_id": episode_id}
            for method, success_map in maps.items():
                paired[method] = success_map.get(episode_id)
            paired_rows.append(paired)
    write_csv(output_dir / "paired_results.csv", paired_rows)
    summary = {"status": "summarized", "num_conditions": len(rows), "rows": rows, "retrieval_queries": len(retrieval_rows)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Phase-Filtered Local Inverse Retrieval — PickCube",
        "",
        "Frozen FM and DefaultEvalRunner are unchanged. Retrieval uses cosine similarity over flattened frozen observation features; same-episode matches are excluded by default.",
        "",
        "| Method | Shift (cm) | Depth | Success | Trials | Rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['shift_cm']:g} | {row['depth']:.1f} | {row.get('successes')} | {row.get('trials')} | {row.get('success_rate')} |")
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Phase random vs phase nearest tests whether local matching fixes random phase-bank sampling.",
        "- Global nearest vs phase nearest tests whether phase filtering adds value beyond local observation retrieval.",
        "- Phase nearest inverse vs phase nearest raw replay uses the same retrieved sample and isolates reconditioning.",
        "- A positive conclusion requires stable multi-seed gains over the native Gaussian baseline; offline retrieval statistics are not sufficient.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    args = parse_args()
    if args.stage in {"build_multimodal", "all"}:
        original_memory_path = args.memory_path
        args.memory_path = original_memory_path
        args.multimodal_memory_path = args.multimodal_memory_path
        print(json.dumps(build_multimodal_memory(args), indent=2), flush=True)
    if args.stage in {"build", "all"}:
        print(json.dumps(build_retrieval_memory(args), indent=2), flush=True)
    if args.stage in {"rollout", "all"}:
        print(json.dumps(run_rollouts(args), indent=2), flush=True)
    if args.stage == "summarize":
        print(json.dumps(summarize(args), indent=2), flush=True)
