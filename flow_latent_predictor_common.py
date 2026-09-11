"""Frozen native FM adapters and reproducible latent predictor utilities."""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def load_flow(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from roboverse_learn.il.utils.flow.flow_matchers import TorchFlowMatcher
    payload = torch.load(args.checkpoint, pickle_module=dill, map_location='cpu', weights_only=False)
    cfg = payload['cfg']
    target = str(cfg.policy_config._target_)
    if target.rsplit('.', 1)[-1] not in ('FlowMatchingUnetImagePolicy', 'FlowMatchingDiTImagePolicy'):
        raise ValueError(f'Only original FM policies supported: {target}')
    policy = hydra.utils.instantiate(cfg.policy_config)
    key = 'ema_model' if cfg.train_config.training_params.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][key], strict=True)
    policy.to(args.device).float().eval().requires_grad_(False)
    if not policy.obs_as_global_cond:
        raise ValueError('Global condition adapter required')
    return policy, TorchFlowMatcher(None), cfg


@torch.no_grad()
def prepare_predictor_context(policy, obs):
    """Encode causal extended history; replace here for task/multimodal context."""
    normalized = policy.normalizer.normalize(obs)
    b, h = next(iter(obs.values())).shape[:2]
    flat = {k: v.reshape(b*h, *v.shape[2:]) for k, v in normalized.items()}
    return policy.obs_encoder(flat).reshape(b, h, -1).detach()


def forward_flow(policy, matcher, z, condition, steps, recompute=True):
    def velocity(x, t, **kwargs):
        # Checkpoint individual vector-field evaluations; unchanged Euler solver.
        def call(xx, tt, cc):
            return policy.model(xx, tt, global_cond=cc)
        if recompute and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(call, x, t, kwargs['global_cond'], use_reentrant=False)
        return call(x, t, kwargs['global_cond'])
    return matcher.sample(velocity, tuple(z.shape), z.device, num_steps=steps, start=z, global_cond=condition)


@torch.no_grad()
def reverse_flow(policy, matcher, action, condition, steps, return_traces=False):
    """Invert a full action chunk into the Flow source space.

    The policy, normalizer and solver are the same ones used by the inversion
    cache.  Keeping this small adapter in the common module lets online code
    invert the chunk that was actually executed without duplicating solver or
    checkpoint loading logic.
    """
    if action.ndim != 3 or tuple(action.shape[1:]) != (policy.horizon, policy.action_dim):
        raise ValueError(
            f"expected action shape [B,{policy.horizon},{policy.action_dim}], got {tuple(action.shape)}"
        )
    if condition.ndim != 2:
        raise ValueError(f"expected flattened condition [B,C], got {tuple(condition.shape)}")
    if return_traces:
        return matcher.reverse_sample(
            policy.model,
            start=action,
            num_steps=steps,
            return_traces=True,
            global_cond=condition,
        )
    return matcher.reverse_sample(
        policy.model,
        start=action,
        num_steps=steps,
        global_cond=condition,
    )


class LatentPredictor(nn.Module):
    def __init__(self, context_shape, latent_shape, hidden_dim, mean, std):
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        self.register_buffer('context_mean', mean)
        self.register_buffer('context_std', std)
        self.net = nn.Sequential(nn.Linear(int(np.prod(context_shape)), hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, int(np.prod(latent_shape))))

    def forward(self, x):
        return self.net(((x-self.context_mean)/self.context_std).flatten(1)).reshape(-1, *self.latent_shape)


def losses(z, target, action, expert, method, lambda_z):
    latent = (z-target).square().mean()
    behavior = (action-expert).square().mean()
    if method == 'latent_mse':
        total = latent
    elif method == 'behavioral':
        total = behavior
    else:
        total = behavior + lambda_z*latent
    return total, latent, behavior


def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cache', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--forward-steps', type=int, default=200)
    p.add_argument('--reverse-steps', type=int, default=200)
    return p


def load_cache(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    shards = [torch.load(path/f, map_location='cpu', weights_only=False) for f in manifest['shards']]
    data = {k: torch.cat([s[k] for s in shards]) for k in shards[0]}
    if len(data['sample_index']) != manifest['samples']:
        raise ValueError('Incomplete cache')
    return data, manifest
