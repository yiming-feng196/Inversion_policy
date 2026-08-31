"""Parameter-free QKV retrieval for an action inverse-state memory.

The current observation is the query (Q), expert observation descriptors are
the keys (K), and the corresponding frozen-Flow inverse action states are the
values (V).  This deliberately has no learned parameters: it is an attention
form of the existing observation-matched memory lookup, so the first
experiment does not introduce a new trainable policy component.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ActionMemoryQKVResult:
    source: torch.Tensor
    top_index: int
    top_scores: torch.Tensor
    top_indices: torch.Tensor
    weights: torch.Tensor
    entropy: float
    effective_k: float


class ActionMemoryQKV:
    """Scaled dot-product attention from an observation query to action memory.

    Image and proprioception are kept as separate Q/K channels and combined
    only in the attention logit.  V is never obtained from the observation;
    it is the stored inverse-flow action state.  This prevents the module from
    silently becoming an action predictor or a new policy.
    """

    def __init__(
        self,
        image_weight: float = 0.8,
        proprio_weight: float = 0.2,
        temperature: float = 0.07,
        top_m: int = 32,
    ) -> None:
        if image_weight < 0 or proprio_weight < 0:
            raise ValueError("QKV modality weights must be non-negative")
        if image_weight + proprio_weight <= 0:
            raise ValueError("At least one QKV modality weight must be positive")
        if temperature <= 0:
            raise ValueError("QKV temperature must be positive")
        self.image_weight = float(image_weight) / (image_weight + proprio_weight)
        self.proprio_weight = float(proprio_weight) / (image_weight + proprio_weight)
        self.temperature = float(temperature)
        self.top_m = max(1, int(top_m))

    @staticmethod
    def _check_pair(query: torch.Tensor, keys: torch.Tensor, name: str) -> None:
        if query.ndim != 1:
            raise ValueError(f"{name} query must be [D], got {tuple(query.shape)}")
        if keys.ndim != 2 or keys.shape[1] != query.shape[0]:
            raise ValueError(
                f"{name} keys must be [N,{query.shape[0]}], got {tuple(keys.shape)}"
            )

    @torch.no_grad()
    def __call__(
        self,
        query_image: torch.Tensor,
        query_proprio: torch.Tensor,
        key_image: torch.Tensor,
        key_proprio: torch.Tensor,
        value_source: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> ActionMemoryQKVResult:
        self._check_pair(query_image, key_image, "image")
        self._check_pair(query_proprio, key_proprio, "proprio")
        if value_source.ndim < 2 or value_source.shape[0] != key_image.shape[0]:
            raise ValueError(
                "value_source must have the same first dimension as memory keys"
            )
        if valid_mask is None:
            valid_mask = torch.ones(key_image.shape[0], dtype=torch.bool, device=key_image.device)
        valid_mask = valid_mask.to(device=key_image.device, dtype=torch.bool)
        valid_indices = torch.where(valid_mask)[0]
        if valid_indices.numel() == 0:
            valid_indices = torch.arange(key_image.shape[0], device=key_image.device)

        q_image = F.normalize(query_image.float(), dim=-1)
        q_proprio = F.normalize(query_proprio.float(), dim=-1)
        k_image = F.normalize(key_image[valid_indices].float(), dim=-1)
        k_proprio = F.normalize(key_proprio[valid_indices].float(), dim=-1)

        # Q = current observation, K = expert observation memory, V = inverse
        # action state.  The modality weights affect only QK^T, never V.
        scores = (
            self.image_weight * (k_image @ q_image)
            + self.proprio_weight * (k_proprio @ q_proprio)
        )
        top_m = min(self.top_m, int(scores.numel()))
        top_scores, top_order = torch.topk(scores, k=top_m, dim=0)
        top_indices = valid_indices[top_order]
        weights = torch.softmax(top_scores / self.temperature, dim=0)
        source = (value_source[top_indices] * weights.reshape(-1, *([1] * (value_source.ndim - 1)))).sum(dim=0)
        entropy = float((-(weights * weights.clamp_min(1e-12).log()).sum() / torch.log(torch.tensor(float(top_m), device=weights.device))).item()) if top_m > 1 else 0.0
        effective_k = float((1.0 / weights.square().sum().clamp_min(1e-12)).item())
        return ActionMemoryQKVResult(
            source=source,
            top_index=int(top_indices[0].item()),
            top_scores=top_scores,
            top_indices=top_indices,
            weights=weights,
            entropy=entropy,
            effective_k=effective_k,
        )


class FlowConditionActionQKV:
    """Single-stream QKV attention in the frozen Flow condition space.

    This is the preferred action-memory variant when online DINO inference is
    undesirable: both Q and K are the exact condition representation consumed
    by the frozen Flow, while V remains an inverse action state.
    """

    def __init__(self, temperature: float = 0.07, top_m: int = 32) -> None:
        if temperature <= 0:
            raise ValueError("QKV temperature must be positive")
        self.temperature = float(temperature)
        self.top_m = max(1, int(top_m))

    @torch.no_grad()
    def __call__(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        value_source: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> ActionMemoryQKVResult:
        if query.ndim != 1 or keys.ndim != 2 or keys.shape[1] != query.shape[0]:
            raise ValueError(
                f"Flow Q/K shapes must be [D] and [N,D], got {tuple(query.shape)} and {tuple(keys.shape)}"
            )
        if value_source.ndim < 2 or value_source.shape[0] != keys.shape[0]:
            raise ValueError("value_source and keys must share the memory dimension")
        if valid_mask is None:
            valid_mask = torch.ones(keys.shape[0], dtype=torch.bool, device=keys.device)
        valid_indices = torch.where(valid_mask.to(device=keys.device, dtype=torch.bool))[0]
        if valid_indices.numel() == 0:
            valid_indices = torch.arange(keys.shape[0], device=keys.device)

        q = F.normalize(query.float(), dim=-1)
        k = F.normalize(keys[valid_indices].float(), dim=-1)
        scores = k @ q
        top_m = min(self.top_m, int(scores.numel()))
        top_scores, top_order = torch.topk(scores, k=top_m, dim=0)
        top_indices = valid_indices[top_order]
        weights = torch.softmax(top_scores / self.temperature, dim=0)
        source = (
            value_source[top_indices]
            * weights.reshape(-1, *([1] * (value_source.ndim - 1)))
        ).sum(dim=0)
        entropy = (
            float(
                (
                    -(weights * weights.clamp_min(1e-12).log()).sum()
                    / torch.log(torch.tensor(float(top_m), device=weights.device))
                ).item()
            )
            if top_m > 1
            else 0.0
        )
        effective_k = float((1.0 / weights.square().sum().clamp_min(1e-12)).item())
        return ActionMemoryQKVResult(
            source=source,
            top_index=int(top_indices[0].item()),
            top_scores=top_scores,
            top_indices=top_indices,
            weights=weights,
            entropy=entropy,
            effective_k=effective_k,
        )
