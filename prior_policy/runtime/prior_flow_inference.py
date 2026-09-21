"""Inference composition: frozen encoder -> Prior Flow -> frozen Action Flow."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn

from roboverse_learn.il.policies.fm.latent_prediction.expert_inversion_prior_flow import (
    ExpertInversionPriorFlow,
)
from roboverse_learn.il.utils.pytorch_util import dict_apply


def load_prior_flow(checkpoint: str | Path, device: str | torch.device):
    """Load the best EMA Prior Flow checkpoint for inference."""
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ExpertInversionPriorFlow(**payload["model_config"])
    model.load_state_dict(payload["ema_model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload


@torch.inference_mode()
def encode_frozen_baseline_condition(action_policy, obs_dict: Dict[str, torch.Tensor]):
    """Use exactly the Action Flow normalizer and observation encoder."""
    action_policy.eval()
    normalized_obs = action_policy.normalizer.normalize(obs_dict)
    value = next(iter(normalized_obs.values()))
    batch_size = value.shape[0]
    obs_steps = action_policy.n_obs_steps
    flat_obs = dict_apply(
        normalized_obs,
        lambda x: x[:, :obs_steps, ...].reshape(-1, *x.shape[2:]),
    )
    features = action_policy.obs_encoder(flat_obs)
    return features.reshape(batch_size, -1)


@torch.inference_mode()
def midpoint_action_flow(action_policy, z_prior, condition, steps=None):
    """Run the original Action Flow vector field with its native midpoint solver."""
    if steps is None:
        steps = int(action_policy.num_inference_steps)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    action_policy.eval()
    z = z_prior
    dt = 1.0 / steps
    for step in range(steps):
        t_start = torch.full(
            (z.shape[0],), step / steps, device=z.device, dtype=z.dtype
        )
        t_mid = torch.full(
            (z.shape[0],), (step + 0.5) / steps, device=z.device, dtype=z.dtype
        )
        velocity_start = action_policy.model(
            z, t_start, local_cond=None, global_cond=condition
        )
        z_mid = z + 0.5 * dt * velocity_start
        velocity_mid = action_policy.model(
            z_mid, t_mid, local_cond=None, global_cond=condition
        )
        z = z + dt * velocity_mid
    return z


@torch.inference_mode()
def predict_action_with_prior(
    action_policy,
    prior_flow: ExpertInversionPriorFlow,
    obs_dict: Dict[str, torch.Tensor],
    prior_steps: int = 16,
    prior_depth: float = 1.0,
    action_steps: int | None = None,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
):
    """Compose the two distinct frozen flows without any gradient graph."""
    if any(parameter.requires_grad for parameter in action_policy.parameters()):
        raise ValueError("Action Flow and visual encoder must be frozen before composition")
    if any(parameter.requires_grad for parameter in prior_flow.parameters()):
        raise ValueError("Prior Flow must be frozen during inference")
    condition = encode_frozen_baseline_condition(action_policy, obs_dict)
    if condition.shape[1] != prior_flow.condition_dim:
        raise ValueError(
            f"condition mismatch: encoder produced {condition.shape[1]}, "
            f"Prior Flow expects {prior_flow.condition_dim}"
        )
    z_prior = prior_flow.sample(
        condition,
        steps=prior_steps,
        end_time=prior_depth,
        noise=noise,
        generator=generator,
    )
    normalized_action = midpoint_action_flow(
        action_policy, z_prior, condition, steps=action_steps
    )
    action_pred = action_policy.normalizer["action"].unnormalize(normalized_action)
    start = action_policy.n_obs_steps - 1
    end = start + action_policy.n_action_steps
    return {
        "action": action_pred[:, start:end],
        "action_pred": action_pred,
        "z_prior": z_prior,
        "condition": condition,
    }


class FrozenPriorActionPolicy(nn.Module):
    """Small adapter exposing the standard ``predict_action`` policy interface."""

    def __init__(
        self,
        action_policy,
        prior_flow,
        prior_steps=16,
        prior_depth=1.0,
        action_steps=None,
    ):
        super().__init__()
        self.action_policy = action_policy.eval().requires_grad_(False)
        self.prior_flow = prior_flow.eval().requires_grad_(False)
        self.prior_steps = int(prior_steps)
        self.prior_depth = float(prior_depth)
        self.action_steps = action_steps

    @torch.inference_mode()
    def predict_action(self, obs_dict):
        return predict_action_with_prior(
            self.action_policy,
            self.prior_flow,
            obs_dict,
            prior_steps=self.prior_steps,
            prior_depth=self.prior_depth,
            action_steps=self.action_steps,
        )
