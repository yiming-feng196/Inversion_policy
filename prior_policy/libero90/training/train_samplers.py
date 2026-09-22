"""Train one source sampler under a fixed, shared cache/optimizer protocol.

Checkpoint selection is ALWAYS the predeclared final optimizer step. Validation
objectives are diagnostics, not action performance and not selection criteria.
No base policy is loaded and no action = expert placeholder exists here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from protocol import (digest_file, learning_rate_multiplier, validate_manifest,
                      validate_resume_manifest, write_json)
from source_models import METHODS, SourceSampler, build_config


def load_cache(cache):
    cache = Path(cache)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Cache extraction must be complete before sampler training")
    data = {}
    for split in ("train", "val"):
        path = cache / manifest["files"][split]
        if digest_file(path) != manifest["file_sha256"][split]:
            raise ValueError(f"Cache file hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in ("condition", "z_star", "expert", "episode", "current")}
        n = len(arrays["condition"])
        if n < 1 or any(len(value) != n for value in arrays.values()):
            raise ValueError(f"Invalid cache lengths: {split}")
        if arrays["condition"].shape != (n, manifest["condition_dim"]):
            raise ValueError("Condition shape differs from manifest")
        shape = (n, *manifest["latent_shape"])
        if arrays["z_star"].shape != shape or arrays["expert"].shape != shape:
            raise ValueError("Source/normalized action shape differs from full-chunk definition")
        for key in ("condition", "z_star", "expert"):
            if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Require finite FP32 {split}/{key}")
        if arrays["episode"].dtype.kind not in "US" or arrays["current"].dtype.kind not in "iu":
            raise ValueError("Episode IDs must be strings; current indices must be integers")
        if arrays["episode"].ndim != 1 or arrays["current"].ndim != 1:
            raise ValueError("Row metadata must be vectors")
        episodes = arrays["episode"].astype(str)
        if set(episodes) != set(manifest["splits"][split]):
            raise ValueError(f"Observed episode IDs differ from declared {split} split")
        row_ids = set(zip(episodes.tolist(), arrays["current"].tolist()))
        if len(row_ids) != n:
            raise ValueError(f"Duplicate action windows in {split}")
        arrays["episode"] = episodes
        # Full expert actions are not inputs or loss targets to the sampler.
        data[split] = arrays
    return data, manifest


def training_condition_statistics(condition):
    values = torch.as_tensor(condition, dtype=torch.float64)
    mean = values.mean(0).float()
    std = values.std(0, unbiased=False).clamp_min(1e-5).float()
    return mean, std


def balanced_validation_indices(episodes, limit, seed=424242):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(episodes == name)).tolist() for name in sorted(set(episodes))]
    chosen = []
    while len(chosen) < min(limit, len(episodes)):
        for group in groups:
            if group:
                chosen.append(group.pop())
            if len(chosen) == min(limit, len(episodes)):
                break
    return np.asarray(chosen, dtype=np.int64)


@torch.no_grad()
def validate(model, data, indices, device, batch_size, prior_steps):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(424242)
    loss_sum, source_error_sum, count = 0., 0., 0
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        condition = torch.as_tensor(data["condition"][rows], device=device)
        target = torch.as_tensor(data["z_star"][rows], device=device)
        noise = torch.randn(target.shape, device=device, generator=rng)
        times = torch.rand((len(rows),), device=device, generator=rng)
        loss = model.objective(condition, target, noise, times)
        generated = model.sample(condition, noise=noise, steps=prior_steps)
        if not torch.isfinite(loss) or not torch.isfinite(generated).all():
            raise FloatingPointError("Nonfinite sampler validation")
        loss_sum += float(loss) * len(rows)
        source_error_sum += float((generated - target).square().flatten(1).mean(1).sqrt().sum())
        count += len(rows)
    return {"validation_objective": loss_sum / count,
            "validation_source_sample_rmse": source_error_sum / count,
            "validation_samples": count,
            "action_decoding_performed": False}


def capture_training_state(optimizer, batch_rng, noise_rng, curve, window_losses,
                           validation_indices, elapsed_seconds, device):
    return {"optimizer": optimizer.state_dict(), "batch_rng": batch_rng.bit_generator.state,
            "noise_rng": noise_rng.get_state(), "python_rng": random.getstate(),
            "numpy_global_rng": np.random.get_state(), "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "curve": list(curve), "window_losses": list(window_losses),
            "validation_indices": validation_indices.copy(), "elapsed_seconds": elapsed_seconds}


def restore_training_state(state, optimizer, batch_rng, noise_rng, device):
    required = {"optimizer", "batch_rng", "noise_rng", "python_rng", "numpy_global_rng",
                "torch_cpu_rng", "torch_cuda_rng", "curve", "window_losses",
                "validation_indices", "elapsed_seconds"}
    if required.difference(state):
        raise ValueError("Checkpoint lacks full optimizer/RNG state; cannot safely resume")
    optimizer.load_state_dict(state["optimizer"])
    batch_rng.bit_generator.state = state["batch_rng"]
    noise_rng.set_state(state["noise_rng"].cpu())
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_global_rng"])
    torch.set_rng_state(state["torch_cpu_rng"].cpu())
    if device.type == "cuda":
        if len(state["torch_cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Visible CUDA device count changed; exact RNG resume refused")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_rng"]])
    elif state["torch_cuda_rng"]:
        raise ValueError("Cannot resume a CUDA RNG stream on CPU")


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="Existing author-backbone repository")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=256)
    parser.add_argument("--prior-steps", type=int, default=16)
    parser.add_argument("--down-dims", default="128,256,512")
    parser.add_argument("--time-embed-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--mlp-layers", type=int, default=3)
    parser.add_argument("--mlp-width", type=int, default=None, help="Default: nearest parameter-count match to source UNet")
    parser.add_argument("--max-allocator-mib", type=int, default=6144)
    parser.add_argument("--resume", action="store_true", help="Resume latest.pt with unchanged protocol and total step budget")
    return parser.parse_args()


def main():
    args = arguments()
    if min(args.steps, args.batch_size, args.eval_every, args.checkpoint_every, args.validation_samples, args.prior_steps) < 1:
        raise ValueError("Training/evaluation budgets must be positive")
    if args.grad_clip <= 0 or args.lr <= 0:
        raise ValueError("Learning rate and gradient clipping threshold must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory; existing runs are never overwritten")
    if args.resume and not ((args.output / "manifest.json").is_file() and (args.output / "latest.pt").is_file()):
        raise FileNotFoundError("Resume requires existing manifest.json and full-state latest.pt")
    if not args.resume:
        args.output.mkdir(parents=True)
    status = {"status": "initializing", "checkpoint_selection": "fixed final optimizer step",
              "expected_steps": args.steps}
    # A refused resume must not modify the previous run's status or logs.
    write_admitted = not args.resume
    if write_admitted:
        write_json(args.output / "status.json", status)
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(args.max_allocator_mib * 1024 ** 2 / total, 1.), device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        data, cache_manifest = load_cache(args.cache)
        mean, std = training_condition_statistics(data["train"]["condition"])
        down_dims = tuple(int(x) for x in args.down_dims.split(","))
        config = build_config(args.method, cache_manifest["condition_dim"], cache_manifest["latent_shape"],
                              args.repo, down_dims=down_dims, time_embed_dim=args.time_embed_dim,
                              kernel_size=args.kernel_size, n_groups=args.n_groups,
                              mlp_layers=args.mlp_layers, mlp_width=args.mlp_width)
        model = SourceSampler(config, mean, std, repo=args.repo).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        reference_parameters = config["reference_flow_parameters"]
        provenance_paths = [Path(__file__), Path(__file__).with_name("source_models.py"),
                            Path(__file__).with_name("protocol.py")]
        author_root = args.repo / "roboverse_learn/il/policies/dp/models/diffusion"
        provenance_paths.extend(author_root / name for name in
                                ("conditional_unet1d.py", "conv1d_components.py", "positional_embedding.py"))
        manifest = {"format_version": 1, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "model_config": config, "parameter_count": parameters,
                    "reference_flow_parameter_count": reference_parameters,
                    "parameter_count_relative_difference": (parameters - reference_parameters) / reference_parameters,
                    "cache_manifest": cache_manifest, "cache_manifest_sha256": digest_file(args.cache / "manifest.json"),
                    "training_samples": len(data["train"]["condition"]), "validation_samples_total": len(data["val"]["condition"]),
                    "condition_normalization_fit_split": "train", "source_target_normalization": "none; native inverted coordinates",
                    "action_normalization": "unchanged frozen Action Flow cache; no refit", "test_split_loaded": False,
                    "checkpoint_selection": "fixed final optimizer step; not validation, latent RMSE, or rollout selected",
                    "source_initialization": "iid standard Gaussian; MLP ignores noise",
                    "source_solver": "left Euler", "training_base_policy_loaded": False,
                    "training_base_policy_gradients": False, "action_decoding_performed": False,
                    "rng": {"batch_indices": args.seed + 10000, "noise_and_time": args.seed + 20000},
                    "torch": torch.__version__, "code": [{"path": str(p), "sha256": digest_file(p)} for p in provenance_paths]}
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        batch_rng = np.random.default_rng(args.seed + 10000)
        noise_rng = torch.Generator(device=device).manual_seed(args.seed + 20000)
        val_indices = balanced_validation_indices(data["val"]["episode"], args.validation_samples)
        curve, window_losses = [], []
        initial_step, elapsed_before, last_metrics = 0, 0., None
        if args.resume:
            original_manifest = json.loads((args.output / "manifest.json").read_text())
            validate_resume_manifest(original_manifest, manifest)
            saved = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=False)
            if saved.get("format") != "q2_source_sampler_v1" or "training_state" not in saved:
                raise ValueError("Legacy/incomplete checkpoint has no resumable training state")
            validate_resume_manifest(original_manifest, saved["manifest"])
            initial_step = int(saved["step"])
            if not 0 <= initial_step <= args.steps:
                raise ValueError("Checkpoint step lies outside the original fixed budget")
            training_state = saved["training_state"]
            np.testing.assert_array_equal(val_indices, training_state["validation_indices"])
            model.load_state_dict(saved["state_dict"], strict=True)
            restore_training_state(training_state, optimizer, batch_rng, noise_rng, device)
            curve, window_losses = list(training_state["curve"]), list(training_state["window_losses"])
            elapsed_before, last_metrics = training_state["elapsed_seconds"], saved.get("metrics")
            manifest = original_manifest
            write_admitted = True
            write_json(args.output / "curve.json", curve)
            if initial_step == args.steps:
                if not (args.output / "final.pt").exists():
                    atomic_checkpoint(args.output / "final.pt", saved)
                status.update(status="completed", completed_steps=initial_step, resumed_from_step=initial_step,
                              final_checkpoint=str(args.output / "final.pt"), elapsed_seconds=elapsed_before)
                write_json(args.output / "status.json", status)
                print(json.dumps(status), flush=True)
                return
            del saved, training_state
            status.update(status="resuming", completed_steps=initial_step, resumed_from_step=initial_step)
            write_json(args.output / "status.json", status)
        else:
            write_json(args.output / "manifest.json", manifest)
            np.save(args.output / "validation_indices.npy", val_indices, allow_pickle=False)

        def checkpoint_payload(step):
            elapsed = elapsed_before + time.monotonic() - started
            return {"format": "q2_source_sampler_v1", "model_config": config,
                    "state_dict": model.state_dict(), "step": step, "seed": args.seed,
                    "manifest": manifest, "metrics": last_metrics,
                    "action_checkpoint_sha256": cache_manifest["action_checkpoint_sha256"],
                    "action_normalizer_sha256": cache_manifest["action_normalizer_sha256"],
                    "checkpoint_selection": "fixed final optimizer step",
                    "training_state": capture_training_state(optimizer, batch_rng, noise_rng, curve,
                        window_losses, val_indices, elapsed, device)}

        # Also recover interruptions before the first periodic checkpoint.
        if not args.resume:
            atomic_checkpoint(args.output / "latest.pt", checkpoint_payload(0))
        for step in range(initial_step + 1, args.steps + 1):
            model.train()
            indices = batch_rng.integers(len(data["train"]["condition"]), size=args.batch_size)
            c = torch.as_tensor(data["train"]["condition"][indices], device=device)
            z = torch.as_tensor(data["train"]["z_star"][indices], device=device)
            noise = torch.randn(z.shape, device=device, generator=noise_rng)
            times = torch.rand((len(z),), device=device, generator=noise_rng)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * learning_rate_multiplier(step, args.steps, args.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            loss = model.objective(c, z, noise, times)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            window_losses.append(float(loss.detach()))
            if step % args.eval_every == 0 or step == args.steps:
                metrics = validate(model, data["val"], val_indices, device, args.batch_size, args.prior_steps)
                last_metrics = metrics
                row = {"step": step, "train_objective": float(np.mean(window_losses)),
                       "elapsed_seconds": elapsed_before + time.monotonic() - started, **metrics}
                curve.append(row)
                window_losses = []
                write_json(args.output / "curve.json", curve)
                print(json.dumps(row, allow_nan=False), flush=True)
            if step % args.checkpoint_every == 0 or step % args.eval_every == 0 or step == args.steps:
                checkpoint = checkpoint_payload(step)
                atomic_checkpoint(args.output / "latest.pt", checkpoint)
                if step == args.steps:
                    atomic_checkpoint(args.output / "final.pt", checkpoint)
                status.update(status="training", completed_steps=step,
                              elapsed_seconds=checkpoint["training_state"]["elapsed_seconds"])
                write_json(args.output / "status.json", status)
                del checkpoint
        status.update(status="completed", completed_steps=args.steps, final_checkpoint=str(args.output / "final.pt"),
                      elapsed_seconds=elapsed_before + time.monotonic() - started)
        if write_admitted:
            write_json(args.output / "status.json", status)
    except Exception as error:
        status.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic() - started)
        if write_admitted:
            write_json(args.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
