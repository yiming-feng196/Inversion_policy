"""Train a conditional Prior Flow from Gaussian noise to expert-inversion latents.

The velocity network intentionally uses the same ConditionalUnet1D architecture
as the frozen PickCube Action Flow.  The important boundary is structural:
training reads cached frozen-encoder conditions and cached expert inversions, so
neither the Action Flow nor its visual encoder is instantiated or differentiated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from roboverse_learn.il.policies.dp.models.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
)
from roboverse_learn.il.utils.ema_model import EMAModel


FORMAT_VERSION = 1


def parse_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not dims or any(x <= 0 for x in dims):
        raise argparse.ArgumentTypeError("down dimensions must be positive integers")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-flow-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)

    # Match the frozen Action Flow architecture by default.
    parser.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    parser.add_argument("--down-dims", type=parse_dims, default=(256, 512, 1024))
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--no-cond-predict-scale", action="store_true")

    # Match the frozen Action Flow optimizer/training scale by default.
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.95, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--diagnostic-every", type=int, default=5)
    parser.add_argument("--diagnostic-samples", type=int, default=128)
    parser.add_argument("--prior-inference-steps", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(value, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_dump(value, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_inversion_cache(path: str | Path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text())
    required = ("condition", "z_star", "split", "episode", "sample_index")
    pieces = {key: [] for key in required}
    for shard_name in manifest["shards"]:
        shard = torch.load(root / shard_name, map_location="cpu", weights_only=True)
        missing = [key for key in required if key not in shard]
        if missing:
            raise KeyError(f"{shard_name} is missing cache fields: {missing}")
        for key in required:
            pieces[key].append(shard[key])
    data = {key: torch.cat(value) for key, value in pieces.items()}
    count = len(data["sample_index"])
    if count != int(manifest["samples"]):
        raise ValueError(f"cache is incomplete: loaded {count}, expected {manifest['samples']}")
    if tuple(data["z_star"].shape[1:]) != tuple(manifest["latent_shape"]):
        raise ValueError("z_star shape disagrees with cache manifest")
    if not torch.isfinite(data["condition"]).all() or not torch.isfinite(data["z_star"]).all():
        raise ValueError("condition or z_star contains non-finite values")
    return data, manifest


class ExpertInversionPriorFlow(nn.Module):
    """Gaussian -> expert-inversion latent conditional flow."""

    def __init__(
        self,
        latent_shape: Iterable[int],
        condition_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Iterable[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.latent_shape = tuple(int(x) for x in latent_shape)
        if len(self.latent_shape) != 2:
            raise ValueError(f"expected (horizon, dim) latent shape, got {self.latent_shape}")
        self.condition_dim = int(condition_dim)
        self.velocity = ConditionalUnet1D(
            input_dim=self.latent_shape[-1],
            local_cond_dim=None,
            global_cond_dim=self.condition_dim,
            diffusion_step_embed_dim=int(diffusion_step_embed_dim),
            down_dims=tuple(int(x) for x in down_dims),
            kernel_size=int(kernel_size),
            n_groups=int(n_groups),
            cond_predict_scale=bool(cond_predict_scale),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor):
        condition = condition.flatten(1)
        if tuple(z_t.shape[1:]) != self.latent_shape:
            raise ValueError(f"expected z shape (*, {self.latent_shape}), got {tuple(z_t.shape)}")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"expected condition dim {self.condition_dim}, got {condition.shape[1]}"
            )
        return self.velocity(z_t, t, local_cond=None, global_cond=condition)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        steps: int = 16,
        end_time: float = 1.0,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Left-Euler integration on the full-path grid up to ``end_time``."""
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if not 0.0 <= end_time <= 1.0:
            raise ValueError(f"end_time must be in [0, 1], got {end_time}")
        condition = condition.flatten(1)
        if noise is None:
            noise = torch.randn(
                (condition.shape[0], *self.latent_shape),
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        z = noise
        dt = 1.0 / steps
        complete_steps = min(steps, int(end_time * steps + 1e-12))
        for step in range(complete_steps):
            t = torch.full(
                (z.shape[0],), step / steps, device=z.device, dtype=z.dtype
            )
            z = z + dt * self(z, t, condition)
        remainder = end_time - complete_steps / steps
        if remainder > 1e-12:
            t = torch.full(
                (z.shape[0],),
                complete_steps / steps,
                device=z.device,
                dtype=z.dtype,
            )
            z = z + remainder * self(z, t, condition)
        return z

    def architecture_config(self) -> dict:
        velocity = self.velocity
        return {
            "latent_shape": list(self.latent_shape),
            "condition_dim": self.condition_dim,
            "diffusion_step_embed_dim": velocity.diffusion_step_encoder[0].dim,
        }


def rectified_flow_loss(
    model: ExpertInversionPriorFlow,
    z_star: torch.Tensor,
    condition: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """E[||v_phi((1-t) eps + t z*, t, c) - (z* - eps)||^2]."""
    epsilon = torch.randn(
        z_star.shape,
        device=z_star.device,
        dtype=z_star.dtype,
        generator=generator,
    )
    t = torch.rand(
        (z_star.shape[0],), device=z_star.device, dtype=z_star.dtype, generator=generator
    )
    t_expanded = t[:, None, None]
    z_t = (1.0 - t_expanded) * epsilon + t_expanded * z_star
    target_velocity = z_star - epsilon
    predicted_velocity = model(z_t, t, condition)
    return F.mse_loss(predicted_velocity, target_velocity)


def batch_indices(indices: torch.Tensor, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


@torch.no_grad()
def validation_loss(model, data, indices, batch_size, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    count = 0
    for batch in batch_indices(indices, batch_size):
        z_star = data["z_star"][batch].to(device, non_blocking=True)
        condition = data["condition"][batch].to(device, non_blocking=True)
        loss = rectified_flow_loss(model, z_star, condition, generator=generator)
        total += float(loss) * len(batch)
        count += len(batch)
    return total / count


@torch.no_grad()
def endpoint_diagnostics(model, data, indices, batch_size, device, seed, steps, count):
    model.eval()
    selected = indices[: min(len(indices), count)]
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = []
    targets = []
    for batch in batch_indices(selected, batch_size):
        condition = data["condition"][batch].to(device, non_blocking=True)
        generated.append(model.sample(condition, steps=steps, generator=generator).cpu())
        targets.append(data["z_star"][batch])
    generated = torch.cat(generated).float()
    targets = torch.cat(targets).float()
    generated_flat = generated.flatten(1)
    targets_flat = targets.flatten(1)
    return {
        "samples": len(selected),
        "prior_inference_steps": steps,
        "paired_latent_mse_diagnostic": float(F.mse_loss(generated, targets)),
        "coordinate_mean_rmse": float(
            F.mse_loss(generated_flat.mean(0), targets_flat.mean(0)).sqrt()
        ),
        "coordinate_std_rmse": float(
            F.mse_loss(
                generated_flat.std(0, unbiased=False), targets_flat.std(0, unbiased=False)
            ).sqrt()
        ),
        "generated_mean": float(generated.mean()),
        "generated_std": float(generated.std(unbiased=False)),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std(unbiased=False)),
    }


def checkpoint_payload(model, ema_model, optimizer, scheduler, ema, args, manifest, epoch, step, best):
    return {
        "format_version": FORMAT_VERSION,
        "epoch": epoch,
        "global_step": step,
        "best_validation_loss": best,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema_optimization_step": ema.optimization_step,
        "ema_decay": ema.decay,
        "config": vars(args),
        "cache_manifest": manifest,
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data, manifest = load_inversion_cache(args.cache)
    train_indices = torch.where(data["split"] == 0)[0]
    validation_indices = torch.where(data["split"] == 1)[0]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("cache must contain non-empty episode-level train and validation splits")
    if set(data["episode"][train_indices].tolist()) & set(
        data["episode"][validation_indices].tolist()
    ):
        raise ValueError("episode leakage between train and validation splits")

    condition_dim = data["condition"].flatten(1).shape[1]
    latent_shape = tuple(data["z_star"].shape[1:])
    model = ExpertInversionPriorFlow(
        latent_shape=latent_shape,
        condition_dim=condition_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=args.down_dims,
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        cond_predict_scale=not args.no_cond_predict_scale,
    ).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMAModel(
        model=ema_model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    natural_steps = math.ceil(len(train_indices) / args.batch_size)
    steps_per_epoch = min(natural_steps, args.max_train_steps) if args.max_train_steps else natural_steps
    total_steps = steps_per_epoch * args.epochs

    def lr_factor(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    latest_path = output / "latest.pt"
    if args.resume:
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        ema_model.load_state_dict(payload["ema_model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        ema.optimization_step = int(payload["ema_optimization_step"])
        ema.decay = float(payload["ema_decay"])
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_validation = float(payload["best_validation_loss"])

    action_flow_hash = sha256(args.action_flow_checkpoint)
    expected_hash = manifest.get("checkpoint_sha256")
    if expected_hash and action_flow_hash != expected_hash:
        raise ValueError(
            "expert inversions were not produced by the supplied Action Flow checkpoint: "
            f"cache={expected_hash}, supplied={action_flow_hash}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_config = {
        "format_version": FORMAT_VERSION,
        "args": vars(args),
        "device": str(device),
        "parameter_count": parameter_count,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "train_episodes": len(set(data["episode"][train_indices].tolist())),
        "validation_episodes": len(set(data["episode"][validation_indices].tolist())),
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "condition_dim": condition_dim,
        "latent_shape": list(latent_shape),
        "action_flow_checkpoint_sha256": action_flow_hash,
        "cache_checkpoint_sha256": expected_hash,
        "training_boundary": {
            "action_flow_loaded": False,
            "visual_encoder_loaded": False,
            "action_flow_backward_calls": 0,
            "condition_source": "cached frozen Action Flow visual encoder embedding",
            "only_trainable_module": "ExpertInversionPriorFlow.velocity",
        },
        "objective": "MSE(v_phi((1-t)*epsilon+t*z_star,t,c), z_star-epsilon)",
        "inference": "epsilon~N(0,I); 16-step left Euler Prior Flow; then frozen Action Flow",
    }
    atomic_json_dump(run_config, output / "config.json")
    print(json.dumps({"event": "start", **run_config}), flush=True)

    train_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    log_path = output / "train.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        permutation_generator = torch.Generator().manual_seed(args.seed + epoch)
        order = train_indices[torch.randperm(len(train_indices), generator=permutation_generator)]
        losses = []
        started = time.time()
        for batch_number, batch in enumerate(batch_indices(order, args.batch_size)):
            if args.max_train_steps and batch_number >= args.max_train_steps:
                break
            z_star = data["z_star"][batch].to(device, non_blocking=True)
            condition = data["condition"][batch].to(device, non_blocking=True)
            loss = rectified_flow_loss(model, z_star, condition, generator=train_generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.step(model)
            losses.append(float(loss.detach()))
            global_step += 1
            if global_step == 1 or global_step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "event": "step",
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "train_loss": losses[-1],
                            "lr": scheduler.get_last_lr()[0],
                        }
                    ),
                    flush=True,
                )

        value = validation_loss(
            ema_model,
            data,
            validation_indices,
            args.batch_size,
            device,
            args.seed + 10_000,
        )
        row = {
            "event": "epoch",
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": sum(losses) / len(losses),
            "validation_loss": value,
            "lr": scheduler.get_last_lr()[0],
            "ema_decay": ema.decay,
            "seconds": time.time() - started,
        }
        if (epoch + 1) % args.diagnostic_every == 0 or epoch + 1 == args.epochs:
            row["endpoint_diagnostics"] = endpoint_diagnostics(
                ema_model,
                data,
                validation_indices,
                args.batch_size,
                device,
                args.seed + 20_000,
                args.prior_inference_steps,
                args.diagnostic_samples,
            )
        with open(log_path, "a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

        if value < best_validation:
            best_validation = value
            atomic_torch_save(
                {
                    "format_version": FORMAT_VERSION,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "validation_loss": value,
                    "ema_model": ema_model.state_dict(),
                    "model_config": {
                        "latent_shape": latent_shape,
                        "condition_dim": condition_dim,
                        "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
                        "down_dims": args.down_dims,
                        "kernel_size": args.kernel_size,
                        "n_groups": args.n_groups,
                        "cond_predict_scale": not args.no_cond_predict_scale,
                    },
                    "prior_inference_steps": args.prior_inference_steps,
                    "action_flow_checkpoint_sha256": action_flow_hash,
                },
                output / "best.pt",
            )
        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    ema,
                    args,
                    manifest,
                    epoch + 1,
                    global_step,
                    best_validation,
                ),
                latest_path,
            )

    atomic_json_dump(
        {
            "completed": True,
            "epochs": args.epochs,
            "global_step": global_step,
            "best_validation_loss": best_validation,
            "best_checkpoint": str(output / "best.pt"),
        },
        output / "completed.json",
    )


if __name__ == "__main__":
    main()
