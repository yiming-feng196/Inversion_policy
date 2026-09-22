"""Dependency-free safeguards for matched source-model experiments."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_manifest(manifest):
    required = {"format_version", "action_checkpoint_sha256", "action_normalizer_sha256",
                "normalizer_fit_split", "executed_slice", "splits", "files",
                "file_sha256", "condition_dim", "latent_shape", "inversion"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Missing cache manifest fields: {sorted(missing)}")
    if manifest["format_version"] != 1 or manifest["normalizer_fit_split"] != "train":
        raise ValueError("Require v1 cache and training-only Action Flow normalization")
    for key in ("action_checkpoint_sha256", "action_normalizer_sha256"):
        value = manifest[key]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid SHA256: {key}")
    names = ("train", "val", "test")
    for name in names:
        values = manifest["splits"].get(name, [])
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate episode names in {name}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(manifest["splits"].get(left, [])) & set(manifest["splits"].get(right, [])):
            raise ValueError(f"Episode overlap: {left}/{right}")
    if not manifest["splits"].get("train") or not manifest["splits"].get("val"):
        raise ValueError("Require nonempty train and validation episode sets")
    shape = manifest["latent_shape"]
    if len(shape) != 2 or min(shape) < 1 or manifest["condition_dim"] < 1:
        raise ValueError("Invalid condition or latent dimensions")
    begin, end = manifest["executed_slice"]
    if not 0 <= begin < end <= shape[0]:
        raise ValueError("Executed slice lies outside the cached full action chunk")
    for name in ("train", "val"):
        relative = Path(manifest["files"][name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Cache files must have safe relative paths")
        if len(manifest["file_sha256"][name]) != 64:
            raise ValueError(f"Invalid file digest for {name}")
    if manifest["inversion"].get("steps", 0) < 1:
        raise ValueError("Cache must record its inverse integration steps")


def mlp_parameter_count(condition_dim, output_dim, hidden_dim, layers=3):
    # Every hidden block: Linear + LayerNorm + SiLU, then an output Linear.
    h = hidden_dim
    return condition_dim * h + (layers - 1) * h * h + 3 * layers * h + output_dim * (h + 1)


def matched_mlp_width(condition_dim, output_dim, target_parameters, layers=3):
    if layers < 2 or target_parameters < 1:
        raise ValueError("Invalid parameter matching request")
    linear = condition_dim + 3 * layers + output_dim
    root = (-linear + math.sqrt(linear * linear + 4 * (layers - 1) *
                                max(target_parameters - output_dim, 0))) / (2 * (layers - 1))
    candidates = {max(1, math.floor(root)), max(1, math.ceil(root))}
    return min(candidates, key=lambda width: abs(
        mlp_parameter_count(condition_dim, output_dim, width, layers) - target_parameters))


def learning_rate_multiplier(step, total_steps, warmup_steps):
    if not 1 <= step <= total_steps:
        raise ValueError("Optimizer step outside fixed training budget")
    warmup = min(step / max(warmup_steps, 1), 1.)
    progress = (step - 1) / max(total_steps, 1)
    return warmup * .5 * (1 + math.cos(math.pi * progress))


def validate_resume_manifest(previous, current):
    """Resume is continuation, not permission to change the fixed protocol."""
    previous_args = {k: v for k, v in previous["args"].items() if k != "resume"}
    current_args = {k: v for k, v in current["args"].items() if k != "resume"}
    if previous_args != current_args:
        changed = sorted(k for k in set(previous_args) | set(current_args)
                         if previous_args.get(k) != current_args.get(k))
        raise ValueError(f"Resume arguments changed: {changed}; use a fresh output directory")
    for key in ("model_config", "cache_manifest_sha256", "code", "torch",
                "condition_normalization_fit_split", "source_target_normalization"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"Resume provenance changed: {key}; use a fresh output directory")
