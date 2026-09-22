"""Frozen pi0 adapter. No action-model parameter is trained or overwritten."""
from __future__ import annotations

import os
import sys
import time
import dataclasses
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HELPERS = Path(os.environ.get("PI0_HELPERS", "/data/jhr/pi0_dsbc_20260913"))
sys.path.insert(0, str(HELPERS))

import einops
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from pi0_flow_ops import _compile_frozen, load_pi0, transform_batch


def make_ops(model):
    def prefix(m, obs):
        obs = model_lib.preprocess_observation(None, obs, train=False, effort_type=m.effort_type)
        tokens, mask, ar = m.embed_prefix(obs)
        (encoded, _), cache = m.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar),
            positions=jnp.cumsum(mask, axis=1) - 1)
        # Pooled frozen VLM features plus the exact normalized current state.
        # No action, timestep-within-demonstration, or episode identity is an input.
        weights = mask.astype(jnp.float32)[..., None]
        pooled = (encoded.astype(jnp.float32) * weights).sum(axis=1) / weights.sum(axis=1)
        condition = jnp.concatenate([pooled, obs.state.astype(jnp.float32)], axis=-1)
        return (obs, mask, cache), condition

    def velocity(m, cached, x, t):
        obs, prefix_mask, kv = cached
        tok, mask, ar = m.embed_suffix(obs, x, jnp.broadcast_to(t, obs.state.shape[0]))
        attn = jnp.concatenate([
            einops.repeat(prefix_mask, "b p -> b s p", s=tok.shape[1]),
            make_attn_mask(mask, ar)], axis=-1)
        pos = prefix_mask.sum(axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (_, output), _ = m.PaliGemma.llm([None, tok], mask=attn, positions=pos, kv_cache=kv)
        return m.action_out_proj(output[:, -m.action_horizon:])

    def integrate(m, cached, x, steps, direction):
        n = steps.astype(jnp.float32)
        dt = direction / n
        def body(k, value):
            t = jnp.where(direction > 0, k / n, 1 - k / n)
            k1 = velocity(m, cached, value, t).astype(jnp.float32)
            k2 = velocity(m, cached, value + dt * k1 / 2, t + dt / 2).astype(jnp.float32)
            k3 = velocity(m, cached, value + dt * k2 / 2, t + dt / 2).astype(jnp.float32)
            k4 = velocity(m, cached, value + dt * k3, t + dt).astype(jnp.float32)
            return value + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        return jax.lax.fori_loop(0, steps, body, x.astype(jnp.float32))

    def stock_euler(m, cached, x):
        # Recursively accumulated FP32 time matches the native pi0 Euler-10.
        dt = -0.1
        def body(carry):
            value, t = carry
            return value + dt * velocity(m, cached, value, t), t + dt
        value, _ = jax.lax.while_loop(lambda a: a[1] >= -dt/2, body, (x, 1.0))
        return value

    return (_compile_frozen(model, prefix), _compile_frozen(model, integrate),
            _compile_frozen(model, stock_euler))


class Runtime:
    def __init__(self, checkpoint, dtype='float32'):
        self.model, self.input_transform, self.output_transform, self.norm_stats = load_pi0(
            "pi0_libero", checkpoint)
        if dtype=='float32':
            from openpi.training import config as config_lib
            # Retain exactly the BF16-rounded weight values from the original
            # checkpoint adapter, while also promoting Gemma/SigLIP activations.
            params=nnx.state(self.model,nnx.Param).to_pure_dict()
            promoted=jax.tree.map(lambda x:x.astype(jnp.float32),params)
            jax.block_until_ready(promoted)
            del params,self.model
            gc.collect()
            cfg=dataclasses.replace(config_lib.get_config('pi0_libero').model,dtype='float32')
            self.model=cfg.load(promoted)
            del promoted
            gc.collect()
            jax.config.update('jax_default_matmul_precision','highest')
        elif dtype!='bfloat16':
            raise ValueError(dtype)
        self.dtype=dtype
        self.prefix, self.rk4, self.decode = make_ops(self.model)
        self.stock = nnx_utils.module_jit(self.model.sample_actions)

    def prepare(self, rows):
        obs, actions, transformed = transform_batch(self.input_transform, rows)
        cached, condition = self.prefix(obs)
        return obs, actions, transformed, cached, condition

    def invert(self, cached, actions, steps):
        return self.rk4(cached, actions, jnp.asarray(steps, jnp.int32), jnp.asarray(1., jnp.float32))

    def high_decode(self, cached, source, steps):
        return self.rk4(cached, source, jnp.asarray(steps, jnp.int32), jnp.asarray(-1., jnp.float32))

    def outputs(self, state, normalized_action):
        return self.output_transform({"state": np.asarray(state), "actions": np.asarray(normalized_action)})["actions"]

    def parity(self, obs, cached):
        key = jax.random.key(123)
        noise = jax.random.normal(key, (obs.state.shape[0], 50, 32))
        native = np.asarray(self.stock(key, obs, num_steps=10))
        custom = np.asarray(self.decode(cached, noise))
        error = metrics(custom, native)
        if error["full_rmse"] > 1e-5:
            raise RuntimeError(f"Stock decoder parity failed: {error}")
        return error


def metrics(a, b):
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return {"full_rmse": float(np.sqrt(np.mean(d*d))),
            "active_rmse": float(np.sqrt(np.mean(d[:, :, :7]**2))),
            "executed_rmse": float(np.sqrt(np.mean(d[:, :10, :7]**2))),
            "max_abs": float(np.abs(d).max())}
