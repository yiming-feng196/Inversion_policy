"""LIBERO adapter for the A2A authors' UNet/DiT velocity backbones.

This is an explicitly new training/analysis harness, not an official LIBERO FM
checkpoint. Original velocity-network files are imported without modification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn
import torchvision


TASKS = ("open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_plate")


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def split_episodes(names, seed=20260915):
    names = list(names)
    order = np.random.default_rng(seed).permutation(len(names))
    a, b = int(.7 * len(names)), int(.8 * len(names))
    return {"train": [names[i] for i in order[:a]],
            "val": [names[i] for i in order[a:b]],
            "test": [names[i] for i in order[b:]]}


def window_rows(length, horizon=16, obs_steps=2, stride=1):
    # Action[obs_steps-1] is the first action following the latest observation.
    return [(cur, cur - obs_steps + 1) for cur in range(obs_steps - 1, length - horizon + obs_steps, stride)]


def norm_params(arrays):
    all_values = np.concatenate(arrays, axis=0).astype(np.float64)
    lower, upper = all_values.min(0), all_values.max(0)
    center = (lower + upper) / 2
    scale = np.where(upper - lower > 1e-6, (upper - lower) / 2, 1.)
    return {"center": center.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist()}


def normalize(x, params):
    return (x - np.asarray(params["center"], np.float32)) / np.asarray(params["scale"], np.float32)


class Corpus:
    def __init__(self, hdf5, split_seed=20260915, normalizer=None, horizon=16, obs_steps=2):
        self.hdf5, self.horizon, self.obs_steps = str(hdf5), horizon, obs_steps
        self.episodes = {}
        with h5py.File(hdf5, "r") as f:
            names = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
            for name in names:
                demo = f["data"][name]
                obs = demo["obs"]
                actions = np.asarray(demo["actions"], np.float32)
                state = np.concatenate([np.asarray(obs["ee_states"]), np.asarray(obs["gripper_states"])], -1).astype(np.float32)
                views = []
                for key in ("agentview_rgb", "eye_in_hand_rgb"):
                    images = np.asarray(obs[key])
                    if images.dtype != np.uint8 or images.shape[1:] != (128, 128, 3):
                        raise ValueError(f"Unexpected image format: {name}/{key}: {images.shape} {images.dtype}")
                    views.append(np.ascontiguousarray(images[:, ::-1, ::-1]))
                if actions.ndim != 2 or actions.shape[1] != 7 or state.shape[1] != 8:
                    raise ValueError(f"Unexpected action/state shapes: {actions.shape}/{state.shape}")
                if not (len(actions) == len(state) == len(views[0]) == len(views[1])):
                    raise ValueError("Observation/action time-axis mismatch")
                if not np.isfinite(actions).all() or not np.isfinite(state).all():
                    raise ValueError("Non-finite dataset")
                self.episodes[name] = {"actions": actions, "state": state, "images": views}
        self.splits = split_episodes(names, split_seed)
        if normalizer is None:
            normalizer = {key: norm_params([self.episodes[n][key] for n in self.splits["train"]])
                          for key in ("actions", "state")}
        self.normalizer = normalizer
        self.rows = {split: [(name, cur, start) for name in chosen
                            for cur, start in window_rows(len(self.episodes[name]["actions"]), horizon, obs_steps)]
                     for split, chosen in self.splits.items()}
        for split, rows in self.rows.items():
            if not rows:
                raise ValueError(f"No valid unpadded windows in {split}")

    def batch(self, rows, device):
        ims, states, actions = [], [], []
        for name, cur, start in rows:
            episode = self.episodes[name]
            # [history, camera, H, W, C], causally ending at cur.
            ims.append(np.stack([x[cur - self.obs_steps + 1:cur + 1] for x in episode["images"]], axis=1))
            states.append(normalize(episode["state"][cur - self.obs_steps + 1:cur + 1], self.normalizer["state"]))
            actions.append(normalize(episode["actions"][start:start + self.horizon], self.normalizer["actions"]))
        images = torch.from_numpy(np.stack(ims)).to(device).permute(0, 1, 2, 5, 3, 4).float().div_(255)
        return (images, torch.as_tensor(np.stack(states), device=device),
                torch.as_tensor(np.stack(actions), device=device))

    def balanced(self, split, count, seed=20260915):
        rng = np.random.default_rng(seed)
        groups = {name: [] for name in self.splits[split]}
        for row in self.rows[split]:
            groups[row[0]].append(row)
        for rows in groups.values():
            rng.shuffle(rows)
        selected = []
        order = list(groups)
        rng.shuffle(order)
        while len(selected) < count:
            added = 0
            for name in order:
                if groups[name]:
                    selected.append(groups[name].pop())
                    added += 1
                    if len(selected) == count:
                        break
            if not added:
                raise ValueError("Insufficient distinct windows")
        return selected


def replace_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            replace_bn(child)


class Policy(nn.Module):
    def __init__(self, repo, architecture, seed=0, obs_steps=2):
        super().__init__()
        sys.path.insert(0, str(repo))
        from roboverse_learn.il.policies.dp.models.diffusion.conditional_unet1d import ConditionalUnet1D
        from roboverse_learn.il.utils.models.flow_net import FlowTransformer
        # Same initialized two-camera visual front-end for a paired architecture seed.
        torch.manual_seed(seed)
        self.vision = nn.ModuleList([torchvision.models.resnet18(weights=None) for _ in range(2)])
        for network in self.vision:
            network.fc = nn.Identity()
            replace_bn(network)
        condition_dim = obs_steps * (2 * 512 + 8)
        torch.manual_seed(seed + 10000)
        if architecture == "unet":
            self.velocity = ConditionalUnet1D(input_dim=7, global_cond_dim=condition_dim,
                diffusion_step_embed_dim=128, down_dims=[256, 512, 1024],
                kernel_size=5, n_groups=8, cond_predict_scale=True)
        elif architecture == "dit":
            self.velocity = FlowTransformer(input_dim=7, condition_dim=condition_dim,
                hidden_dim=512, output_dim=7, num_layers=4, num_heads=8,
                mlp_ratio=4, dropout=.1, time_embed_dim=256)
        else:
            raise ValueError(architecture)
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def condition(self, images, state):
        b, h = state.shape[:2]
        features = []
        for i, encoder in enumerate(self.vision):
            frame = images[:, :, i].reshape(b * h, 3, 128, 128)
            features.append(encoder((frame - self.rgb_mean) / self.rgb_std).reshape(b, h, -1))
        return torch.cat([*features, state], -1).flatten(1)

    def field(self, x, t, c):
        return self.velocity(x, t, global_cond=c)

    def forward(self, images, state, x, t):
        return self.field(x, t, self.condition(images, state))


def integrate(policy, condition, x, steps, direction=1, solver="rk4"):
    if steps < 1 or direction not in (-1, 1) or solver not in ("euler", "midpoint", "rk4"):
        raise ValueError("Invalid integration configuration")
    dt = direction / steps
    for k in range(steps):
        t = k / steps if direction == 1 else 1 - k / steps
        def field(v, at):
            return policy.field(v, torch.full((len(v),), at, device=v.device), condition)
        v1 = field(x, t)
        if solver == "euler":
            x = x + dt * v1
        else:
            v2 = field(x + dt * v1 / 2, t + dt / 2)
            if solver == "midpoint":
                x = x + dt * v2
            else:
                v3 = field(x + dt * v2 / 2, t + dt / 2)
                v4 = field(x + dt * v3, t + dt)
                x = x + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    return x


def fm_loss(model, batch, generator):
    images, state, action = batch
    noise = torch.randn(action.shape, generator=generator, device=action.device)
    t = torch.rand((len(action),), generator=generator, device=action.device)
    x = (1-t[:, None, None]) * noise + t[:, None, None] * action
    prediction = model(images, state, x, t)
    return (prediction - (action-noise)).square().mean()


@torch.no_grad()
def validate(model, corpus, device, size=32, samples=256):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(424242)
    rows = corpus.balanced("val", min(samples, len(corpus.rows["val"])), 424242)
    losses = []
    for start in range(0, len(rows), size):
        selected = rows[start:start+size]
        losses.append((float(fm_loss(model, corpus.batch(selected, device), rng)), len(selected)))
    return sum(loss*n for loss, n in losses) / sum(n for _, n in losses)


def provenance(repo):
    paths = [Path(__file__), Path(repo) / "roboverse_learn/il/utils/models/flow_net.py",
             Path(repo) / "roboverse_learn/il/utils/models/layers.py",
             Path(repo) / "roboverse_learn/il/policies/dp/models/diffusion/conditional_unet1d.py",
             Path(repo) / "roboverse_learn/il/policies/dp/models/diffusion/conv1d_components.py",
             Path(repo) / "roboverse_learn/il/policies/dp/models/diffusion/positional_embedding.py"]
    return [{"path": str(p), "sha256": digest_file(p)} for p in paths]


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a fresh directory; existing experiments are never overwritten")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    corpus = Corpus(args.hdf5)
    model = Policy(args.repo, args.arch, args.seed).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    rng = np.random.default_rng(args.seed + 20000)
    torch_rng = torch.Generator(device=args.device).manual_seed(args.seed + 30000)
    manifest = {"status": "running", "args": vars(args), "splits": corpus.splits,
        "normalizer": corpus.normalizer, "normalizer_fit_split": "train",
        "data_sha256": digest_file(args.hdf5), "episode_lengths": {k:len(v["actions"]) for k,v in corpus.episodes.items()},
        "windows": {k:len(v) for k,v in corpus.rows.items()}, "code": provenance(args.repo),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "protocol": "FP32; author velocity classes; from-scratch ResNet18+GroupNorm per camera; Gaussian source; no clipping/moment matching; history=2, horizon=16, executed=1:9; official 128px images rotated 180deg; no padding",
        "scope": "Custom LIBERO training adapter around author backbones; not author checkpoint or controlled parameter-count comparison"}
    write_json(output / "manifest.json", manifest)
    initial = validate(model, corpus, args.device, args.batch_size, args.validation_samples)
    best = initial
    print(json.dumps({"step": 0, "val_fm": initial, "parameters":manifest["parameter_count"]}), flush=True)
    started = time.monotonic()
    running = []
    curve = []
    for step in range(1, args.steps+1):
        model.train()
        ix = rng.integers(len(corpus.rows["train"]), size=args.batch_size)
        batch = corpus.batch([corpus.rows["train"][i] for i in ix], args.device)
        lr_mult = min(step / 200, 1.) * .5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_mult
        optimizer.zero_grad(set_to_none=True)
        loss = fm_loss(model, batch, torch_rng)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradients")
        optimizer.step()
        running.append(float(loss.detach()))
        if step % 50 == 0:
            print(json.dumps({"step": step, "train_fm": float(np.mean(running[-50:])),
                "seconds": time.monotonic()-started}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = validate(model, corpus, args.device, args.batch_size, args.validation_samples)
            row = {"step":step, "train_fm":float(np.mean(running)), "val_fm":val,
                   "seconds":time.monotonic()-started}
            curve.append(row)
            running = []
            checkpoint = {"state_dict":model.state_dict(), "manifest":manifest, "step":step, "val_fm":val}
            if val < best or not (output / "best.pt").exists():
                best = val
                torch.save(checkpoint, output / "best.pt")
            torch.save(checkpoint, output / "last.pt")
            write_json(output / "curve.json", curve)
            print(json.dumps(row), flush=True)
    manifest.update(status="complete", best_val_fm=best, initial_val_fm=initial, elapsed_seconds=time.monotonic()-started)
    write_json(output / "manifest.json", manifest)


def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a,np.float64)-b)**2, axis=(1,2))).mean())


def error_summary(a, b, episodes):
    difference = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    per_sample = np.sqrt(np.mean(difference.reshape(len(difference), -1)**2, axis=1))
    names = np.asarray(episodes)
    per_episode = {name: float(per_sample[names == name].mean()) for name in np.unique(names)}
    values = np.asarray(list(per_episode.values()))
    bootstrap = np.random.default_rng(20260916).choice(values, (1000, len(values)), replace=True).mean(1)
    return {"mean_per_sample_rmse": float(per_sample.mean()),
            "global_rmse": float(np.sqrt(np.mean(difference**2))),
            "p95_per_sample_rmse": float(np.quantile(per_sample, .95)),
            "max_per_sample_rmse": float(per_sample.max()),
            "per_sample_rmse": per_sample.tolist(), "per_episode_mean_rmse": per_episode,
            "episode_balanced_mean": float(values.mean()),
            "episode_bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "uncertainty_scope": "Descriptive episode bootstrap; not across independently trained models"}


@torch.no_grad()
def audit(args):
    from q1_statistics import source_diagnostics
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = ckpt["manifest"]
    for source in manifest["code"][1:]:
        if digest_file(source["path"]) != source["sha256"]:
            raise ValueError(f"Author backbone changed since training: {source['path']}")
    corpus = Corpus(manifest["args"]["hdf5"], normalizer=manifest["normalizer"])
    if corpus.splits != manifest["splits"]:
        raise ValueError("Split mismatch")
    if digest_file(corpus.hdf5) != manifest["data_sha256"]:
        raise ValueError("Dataset changed since training")
    model = Policy(args.repo, manifest["args"]["arch"], manifest["args"]["seed"]).to(args.device).eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    rows = corpus.balanced(args.split, args.samples, args.seed)
    ns = sorted(set(int(x) for x in args.inverse_steps.split(",")))
    if not ns or min(ns) < 1 or args.reference_steps < 2:
        raise ValueError("Invalid inverse/reference steps")
    noise = np.random.default_rng(args.seed+1).standard_normal((args.samples,16,7)).astype(np.float32)
    result = {"checkpoint":args.checkpoint, "checkpoint_sha256":digest_file(args.checkpoint), "args":vars(args),
              "status":"running", "split":args.split, "episodes":len(set(x[0] for x in rows)), "rows":rows,
              "reference":"Fixed RK4 reference, not exact oracle", "metrics":"Mean per-sample RMSE", "cases":{}, "code":provenance(args.repo),
              "training_manifest":manifest, "checkpoint_step":ckpt["step"], "checkpoint_val_fm":ckpt["val_fm"],
              "sampling":"Episode-balanced, no replacement, no terminal padding; windows within episodes are correlated"}
    arrays = {n:{key:[] for key in ("known_gaussian","recovered_gaussian","expert_inverse","cycle_expert","expert_action","midpoint10", "native_action")} for n in ns}
    controls = {key:[] for key in ("native_half", "native_full")}
    if args.region_control:
        controls.update({key:[] for key in ("region_known", "region_recovered", "region_action_full", "region_action_half")})
    write_json(output / "summary.json", result)
    for start in range(0, args.samples, args.batch_size):
        selected = rows[start:start+args.batch_size]
        images, state, action = corpus.batch(selected,args.device)
        cond = model.condition(images,state)
        z = torch.as_tensor(noise[start:start+len(selected)], device=args.device)
        native = integrate(model,cond,z,args.reference_steps)
        controls["native_full"].append(native.cpu().numpy())
        controls["native_half"].append(integrate(model,cond,z,args.reference_steps//2).cpu().numpy())
        for n in ns:
            recovered = integrate(model,cond,native,n,-1,solver=args.inverse_solver)
            inv = integrate(model,cond,action,n,-1,solver=args.inverse_solver)
            cycle = integrate(model,cond,inv,args.reference_steps)
            deployed = integrate(model,cond,inv,10,solver="midpoint")
            for key,value in zip(arrays[n],(z,recovered,inv,cycle,action,deployed,native)):
                arrays[n][key].append(value.cpu().numpy())
            batch_arrays = {k:v[-1] for k,v in arrays[n].items()}
            np.savez_compressed(output / f"batch_{start:04d}_{args.inverse_solver}_{n}.npz", **batch_arrays)
            if not all(np.isfinite(v).all() for v in batch_arrays.values()):
                result.update(status="nonfinite_numerical_failure", failed_batch=start, failed_inverse_steps=n)
                write_json(output / "summary.json", result)
                raise FloatingPointError("Non-finite inversion coordinates; failing arrays preserved")
        if args.region_control:
            # A known-source control near the expert inverse, not a claim that
            # the expert inverse is ground truth. Independent of Gaussian control.
            anchor = integrate(model,cond,action,args.reference_steps,-1)
            region_known = anchor + .1 * z
            region_full = integrate(model,cond,region_known,args.reference_steps)
            region_half = integrate(model,cond,region_known,args.reference_steps//2)
            region_recovered = integrate(model,cond,region_full,args.reference_steps,-1)
            for key, value in zip(("region_known", "region_recovered", "region_action_full", "region_action_half"),
                                  (region_known,region_recovered,region_full,region_half)):
                controls[key].append(value.cpu().numpy())
        result["completed_samples"] = start+len(selected)
        write_json(output / "summary.json",result)
        print(json.dumps({"stage":"audit_batch", "completed":result["completed_samples"]}),flush=True)
    previous = None
    episode_names = np.asarray([x[0] for x in rows])
    for n in ns:
        a = {k:np.concatenate(v) for k,v in arrays[n].items()}
        if not all(np.isfinite(v).all() for v in a.values()):
            raise FloatingPointError("Non-finite inversion coordinates")
        metrics = {"source_recovery":rmse(a["known_gaussian"],a["recovered_gaussian"]),
                   "expert_reconstruction":rmse(a["cycle_expert"],a["expert_action"]),
                   "midpoint10_reconstruction":rmse(a["midpoint10"],a["expert_action"]),
                   "source_change":None if previous is None else rmse(a["expert_inverse"],previous)}
        metrics["details"] = {label:error_summary(a[left],a[right],episode_names) for label,left,right in (
            ("source_recovery","known_gaussian","recovered_gaussian"),
            ("expert_reconstruction","cycle_expert","expert_action"),
            ("midpoint10_reconstruction","midpoint10","expert_action"),
            ("native_vs_paired_expert_not_a_success_metric","native_action","expert_action"))}
        metrics["executed_slice_reconstruction"] = error_summary(a["cycle_expert"][:,1:9],a["expert_action"][:,1:9],episode_names)
        diagnostics = source_diagnostics({k:a[k] for k in ("known_gaussian","recovered_gaussian","expert_inverse")})
        np.savez_compressed(output / f"{args.inverse_solver}_{n}.npz", **a, episode_name=episode_names,current=np.array([x[1] for x in rows]))
        write_json(output / f"distribution_{n}.json",diagnostics)
        result["cases"][str(n)] = metrics
        previous = a["expert_inverse"]
    controls = {k:np.concatenate(v) for k,v in controls.items()}
    np.savez_compressed(output / "reference_controls.npz", **controls, episode_name=episode_names)
    result["reference_half_vs_full"] = error_summary(controls["native_half"],controls["native_full"],episode_names)
    if args.region_control:
        result["region_source_recovery"] = error_summary(controls["region_recovered"],controls["region_known"],episode_names)
        result["region_reference_half_vs_full"] = error_summary(controls["region_action_half"],controls["region_action_full"],episode_names)
    # Predeclared numerical gate for deciding whether to enlarge the sample set.
    # This does not assess policy quality, Gaussianity, or task success.
    last = result["cases"][str(max(ns))]
    checks = {"source_recovery":last["source_recovery"], "expert_reconstruction":last["expert_reconstruction"],
              "reference_half_vs_full":result["reference_half_vs_full"]["mean_per_sample_rmse"]}
    if last["source_change"] is not None:
        checks["source_change"] = last["source_change"]
    if args.region_control:
        checks["region_source_recovery"] = result["region_source_recovery"]["mean_per_sample_rmse"]
    result["numerical_gate"] = {"threshold":.001, "values":checks,
                               "passed":all(v < .001 for v in checks.values()),
                               "meaning":"Mean normalized-coordinate errors only; inspect sample tails separately"}
    result["status"]="complete"
    write_json(output / "summary.json",result)
    print(json.dumps({"status":"complete", "output":str(output), "numerical_gate":result["numerical_gate"]}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("train","audit"))
    p.add_argument("--repo",required=True)
    p.add_argument("--hdf5")
    p.add_argument("--arch",choices=("unet","dit"))
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=10000)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--eval-every",type=int,default=500)
    p.add_argument("--validation-samples",type=int,default=256)
    p.add_argument("--checkpoint")
    p.add_argument("--split",choices=("train","val","test"),default="test")
    p.add_argument("--samples",type=int,default=128)
    p.add_argument("--inverse-steps",default="64,128,256")
    p.add_argument("--inverse-solver",choices=("euler","midpoint","rk4"),default="rk4")
    p.add_argument("--reference-steps",type=int,default=256)
    p.add_argument("--region-control",action="store_true")
    args=p.parse_args()
    train(args) if args.mode=="train" else audit(args)

if __name__=="__main__":
    main()
