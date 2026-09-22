"""Matched source samplers using the existing author ConditionalUnet1D.

The base visual encoder and action decoder are NOT loaded during training.
Each method sees exactly the same cached frozen conditions and full sources.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import nn

from protocol import matched_mlp_width


METHODS = ("cflow", "uflow", "mlp", "cgaussian")


def author_velocity(repo, condition_dim, action_dim, config):
    sys.path.insert(0, str(Path(repo).resolve()))
    from roboverse_learn.il.policies.dp.models.diffusion.conditional_unet1d import ConditionalUnet1D
    return ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=condition_dim,
        diffusion_step_embed_dim=config["time_embed_dim"], down_dims=config["down_dims"],
        kernel_size=config["kernel_size"], n_groups=config["n_groups"],
        cond_predict_scale=True)


class SourceSampler(nn.Module):
    def __init__(self, config, condition_mean, condition_std, repo=None, velocity=None):
        super().__init__()
        self.config = dict(config)
        self.method = config["method"]
        if self.method not in METHODS:
            raise ValueError(self.method)
        self.latent_shape = tuple(config["latent_shape"])
        self.condition_dim = int(config["condition_dim"])
        self.register_buffer("condition_mean", condition_mean.detach().clone().float())
        self.register_buffer("condition_std", condition_std.detach().clone().float())
        if self.condition_mean.shape != (self.condition_dim,) or self.condition_std.shape != (self.condition_dim,):
            raise ValueError("Condition statistics have incorrect shape")
        if not torch.isfinite(self.condition_mean).all() or not torch.isfinite(self.condition_std).all() or (self.condition_std <= 0).any():
            raise ValueError("Invalid training condition normalization")
        if self.method in ("cflow", "uflow"):
            self.velocity = velocity if velocity is not None else author_velocity(
                repo, self.condition_dim, self.latent_shape[-1], config)
        else:
            width = config["mlp_width"]
            layers = []
            dim = self.condition_dim
            for _ in range(config["mlp_layers"]):
                layers.extend([nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU()])
                dim = width
            output_dim = math.prod(self.latent_shape) * (2 if self.method == "cgaussian" else 1)
            layers.append(nn.Linear(dim, output_dim))
            self.regressor = nn.Sequential(*layers)

    def normalize_condition(self, condition):
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError("Require flattened frozen conditions [batch, C]")
        if self.method == "uflow":
            # Identical network capacity but no access to the current condition.
            return torch.zeros_like(condition)
        return (condition - self.condition_mean) / self.condition_std

    def point_or_gaussian(self, condition):
        output = self.regressor(self.normalize_condition(condition))
        if self.method == "mlp":
            return output.reshape(-1, *self.latent_shape), None
        mean, log_std = output.chunk(2, dim=-1)
        return (mean.reshape(-1, *self.latent_shape),
                log_std.clamp(-5., 2.).reshape(-1, *self.latent_shape))

    def objective(self, condition, target, noise, times):
        if target.shape != noise.shape or tuple(target.shape[1:]) != self.latent_shape:
            raise ValueError("All methods must predict the same complete source sequence")
        if self.method in ("cflow", "uflow"):
            tau = times.reshape(-1, 1, 1)
            interpolant = (1 - tau) * noise + tau * target
            prediction = self.velocity(interpolant, times, global_cond=self.normalize_condition(condition))
            return (prediction - (target - noise)).square().mean()
        mean, log_std = self.point_or_gaussian(condition)
        if self.method == "mlp":
            return (mean - target).square().mean()
        return (.5 * ((target - mean) * (-log_std).exp()).square() + log_std).mean()

    @torch.no_grad()
    def sample(self, condition, *, noise=None, steps=16, generator=None):
        if steps < 1:
            raise ValueError("Prior integration steps must be positive")
        expected = (len(condition), *self.latent_shape)
        if self.method == "mlp":
            return self.point_or_gaussian(condition)[0]
        if noise is None:
            noise = torch.randn(expected, generator=generator, device=condition.device, dtype=condition.dtype)
        if tuple(noise.shape) != expected:
            raise ValueError("External common random noise has incorrect shape")
        if self.method == "cgaussian":
            mean, log_std = self.point_or_gaussian(condition)
            return mean + log_std.exp() * noise
        # Left Euler, consistent with the existing source-Flow design. Do not mutate noise.
        x = noise.clone()
        normalized = self.normalize_condition(condition)
        for k in range(steps):
            times = torch.full((len(condition),), k / steps, device=x.device, dtype=x.dtype)
            x = x + self.velocity(x, times, global_cond=normalized) / steps
        return x


def build_config(method, condition_dim, latent_shape, repo, down_dims=(128, 256, 512),
                 time_embed_dim=128, kernel_size=5, n_groups=8, mlp_layers=3,
                 mlp_width=None):
    config = dict(method=method, condition_dim=int(condition_dim), latent_shape=list(latent_shape),
                  down_dims=list(down_dims), time_embed_dim=time_embed_dim,
                  kernel_size=kernel_size, n_groups=n_groups, mlp_layers=mlp_layers)
    # All methods are compared to the exact same author-UNet parameter count.
    # Fork the RNG so counting a reference does not change model initialization.
    with torch.random.fork_rng(devices=[]):
        reference = author_velocity(repo, condition_dim, latent_shape[-1], config)
        reference_count = sum(p.numel() for p in reference.parameters())
    del reference
    if method in ("mlp", "cgaussian"):
        out = math.prod(latent_shape) * (2 if method == "cgaussian" else 1)
        config["mlp_width"] = mlp_width or matched_mlp_width(condition_dim, out, reference_count, mlp_layers)
    config["reference_flow_parameters"] = reference_count
    return config


def load_sampler(checkpoint_path, repo, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "q2_source_sampler_v1":
        raise ValueError("Wrong sampler checkpoint format")
    state = checkpoint["state_dict"]
    model = SourceSampler(checkpoint["model_config"], state["condition_mean"],
                          state["condition_std"], repo=repo)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint
