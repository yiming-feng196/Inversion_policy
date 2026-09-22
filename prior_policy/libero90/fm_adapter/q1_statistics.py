"""CPU-only diagnostics for action-noise sources in their native coordinates.

Every group must contain the same number of samples and flattened features.
Callers are responsible for removing inactive/padded action coordinates first.
No group is standardized or rescaled. The sliced Wasserstein distances share
unit directions and one independent N(0, I) reference, including the 20-replicate
default finite-sample Gaussian null. These diagnostics do not prove Gaussianity.

Example:
    python q1_statistics.py --input sources.npz --outputdir results

The NPZ must contain known_gaussian, recovered_gaussian, and expert_inverse.
Only NumPy is required; a figure is also produced when matplotlib is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


SOURCE_KEYS = ("known_gaussian", "recovered_gaussian", "expert_inverse")
DEFAULT_SEED = 20260915


def _flatten_sources(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("At least one source array is required.")
    flattened = {}
    expected_shape = None
    for name, array in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Source names must be nonempty strings.")
        raw = np.asarray(array)
        if raw.ndim < 2 or raw.shape[0] < 1 or raw.size == 0:
            raise ValueError(f"{name}: expected a nonempty (N, ...) array with feature axes.")
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise ValueError(f"{name}: source values must be real numbers.")
        flat = np.asarray(raw, dtype=np.float64).reshape(raw.shape[0], -1)
        if not np.isfinite(flat).all():
            raise ValueError(f"{name}: source contains nonfinite values.")
        if expected_shape is None:
            expected_shape = flat.shape
        elif flat.shape != expected_shape:
            raise ValueError(
                f"{name}: flattened shape {flat.shape} does not match {expected_shape}; "
                "all sources must have equal N and feature dimension."
            )
        flattened[name] = flat
    return flattened


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _summary(values: np.ndarray) -> dict[str, float]:
    # np.std can report a tiny nonzero value for repeated decimals such as 0.1
    # when their arithmetic mean rounds differently from the stored value.
    if np.all(values == values.flat[0]):
        value = float(values.flat[0])
        return {"mean": value, "std": 0.0, "min": value, "max": value}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _center_features(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center using differences from an anchor, preserving exact constant axes."""
    offsets = array - array[:1]
    mean_offset = offsets.mean(axis=0, keepdims=True)
    return array[:1] + mean_offset, offsets - mean_offset


def _covariance_spectrum(array: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Return covariance eigenvalues from data SVD, never a D-by-D covariance."""
    _, centered = _center_features(array)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    if array.shape[0] == 1 or singular_values[0] == 0:
        return np.zeros_like(singular_values), 0, 0.0
    tolerance = singular_values[0] * max(array.shape) * np.finfo(np.float64).eps
    positive = singular_values > tolerance
    spectrum = np.zeros_like(singular_values)
    spectrum[positive] = singular_values[positive] ** 2 / (array.shape[0] - 1)
    # Scaling the singular values first avoids an unnecessarily large trace in
    # the entropy calculation. Zero modes contribute zero entropy.
    weights = (singular_values[positive] / singular_values[0]) ** 2
    probabilities = weights / weights.sum()
    effective_rank = np.exp(-np.sum(probabilities * np.log(probabilities)))
    return spectrum, int(positive.sum()), float(effective_rank)


def _sliced_w2(sorted_a: np.ndarray, sorted_b: np.ndarray) -> float:
    """sqrt(mean_direction(mean_sample((sorted projection difference)^2)))."""
    return float(np.sqrt(np.mean((sorted_a - sorted_b) ** 2)))


def source_diagnostics(
    arrays: Mapping[str, np.ndarray],
    seed: int = DEFAULT_SEED,
    *,
    n_directions: int = 128,
    gaussian_null_replicates: int = 20,
) -> dict:
    """Return a JSON-safe report for equal-N, equal-D sample groups.

    Features are flattened after the sample axis. Means, standard deviations,
    and radii use population standard deviations (ddof=0); covariance uses
    ddof=1. Effective rank is exp(entropy of normalized covariance eigenvalues),
    defined as zero for zero covariance. The spectrum omits structural trailing
    zero modes beyond min(N, D), whose count is reported separately.

    Gaussian null draws are independent of the shared reference and one another.
    Their distances all use the same directions and reference as the data.
    Quantiles describe this finite Monte Carlo sample, not a significance test.
    """
    flat = _flatten_sources(arrays)
    n_directions = _positive_integer(n_directions, "n_directions")
    gaussian_null_replicates = _positive_integer(gaussian_null_replicates, "gaussian_null_replicates")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    seed = int(seed)
    n_samples, n_features = next(iter(flat.values())).shape
    direction_seed, reference_seed, null_seed = np.random.SeedSequence(seed).spawn(3)
    direction_rng = np.random.default_rng(direction_seed)
    directions = direction_rng.standard_normal((n_features, n_directions))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    reference = np.random.default_rng(reference_seed).standard_normal((n_samples, n_features))
    reference_projection = np.sort(reference @ directions, axis=0)

    groups = {}
    projections = {}
    for name, array in flat.items():
        coordinate_center, centered = _center_features(array)
        coordinate_means = coordinate_center[0]
        coordinate_stds = np.sqrt(np.mean(centered ** 2, axis=0))
        squared_radii = np.sum(array ** 2, axis=1)
        spectrum, rank, effective_rank = _covariance_spectrum(array)
        global_summary = _summary(array)
        groups[name] = {
            "mean": global_summary["mean"],
            "std": global_summary["std"],
            "coordinate_mean_summary": {
                **_summary(coordinate_means),
                "rms": float(np.sqrt(np.mean(coordinate_means ** 2))),
            },
            "coordinate_std_summary": _summary(coordinate_stds),
            "l2_norm": _summary(np.sqrt(squared_radii)),
            "squared_l2_norm": _summary(squared_radii),
            "covariance_trace": float(spectrum.sum()),
            "covariance_eigenvalues": spectrum.tolist(),
            "omitted_trailing_zero_eigenvalues": int(n_features - len(spectrum)),
            "numerical_covariance_rank": rank,
            "effective_rank": effective_rank,
        }
        projections[name] = np.sort(array @ directions, axis=0)

    pairwise = {name: {} for name in flat}
    names = list(flat)
    for i, name in enumerate(names):
        pairwise[name][name] = 0.0
        for other in names[i + 1:]:
            distance = _sliced_w2(projections[name], projections[other])
            pairwise[name][other] = distance
            pairwise[other][name] = distance

    null_rng = np.random.default_rng(null_seed)
    null_distances = []
    for _ in range(gaussian_null_replicates):
        independent_gaussian = null_rng.standard_normal((n_samples, n_features))
        null_projection = np.sort(independent_gaussian @ directions, axis=0)
        null_distances.append(_sliced_w2(null_projection, reference_projection))
    null_distances = np.asarray(null_distances)

    report = {
        "schema_version": 1,
        "seed": seed,
        "n_samples_per_group": n_samples,
        "n_flattened_features": n_features,
        "coordinate_policy": "native common coordinates for distances and radii; no per-group standardization or rescaling; covariance is centered by definition",
        "std_ddof": 0,
        "covariance_ddof": 1,
        "effective_rank_definition": "exp(entropy(normalized covariance eigenvalues)); zero for zero covariance",
        "groups": groups,
        "sliced_w2": {
            "n_directions": n_directions,
            "directions": "shared random unit-L2 directions in native feature coordinates",
            "definition": "sqrt(mean over directions and samples of squared differences of sorted projections)",
            "reference": "one shared independent standard Gaussian with the same N and D",
            "vs_shared_gaussian": {
                name: _sliced_w2(projection, reference_projection)
                for name, projection in projections.items()
            },
            "pairwise": pairwise,
            "gaussian_null": {
                "replicates": gaussian_null_replicates,
                "distances": null_distances.tolist(),
                **_summary(null_distances),
                "q025": float(np.quantile(null_distances, 0.025)),
                "median": float(np.median(null_distances)),
                "q975": float(np.quantile(null_distances, 0.975)),
                "interpretation": "finite-N Gaussian-vs-Gaussian Monte Carlo reference using the same fixed reference and directions; not a hypothesis test",
            },
        },
    }
    # Catch finite inputs whose magnitude causes overflow in derived statistics.
    # Returning NaN or infinity would silently create nonstandard JSON.
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as exc:
        raise ValueError("Source magnitude overflowed the diagnostics; use representable native units.") from exc
    return report


def _shared_pca(flat: Mapping[str, np.ndarray]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Fit exactly one center and basis to all groups pooled together."""
    pooled = np.concatenate(list(flat.values()), axis=0)
    center, centered = _center_features(pooled)
    _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    basis = basis[:2]
    scores = {}
    for name, array in flat.items():
        projected = (array - center) @ basis.T
        if projected.shape[1] == 1:
            projected = np.pad(projected, ((0, 0), (0, 1)))
        scores[name] = projected
    squared = singular_values ** 2
    explained = squared[:2] / squared.sum() if squared.sum() > 0 else np.zeros(min(2, len(squared)))
    explained = np.pad(explained, (0, 2 - len(explained)))
    return scores, center, basis, explained


def plot_source_diagnostics(arrays: Mapping[str, np.ndarray], path: str | Path) -> str:
    """Save a shared-PCA, covariance-spectrum, and native-radius figure.

    `path` names the output image (for example source_diagnostics.png). Requires
    matplotlib, imported only on use. All groups share one pooled center and PCA
    basis, common histogram bins, and native units. Covariance alone is centered
    per group by definition; the PCA scatter and radii retain group mean shifts.
    """
    flat = _flatten_sources(arrays)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires the optional matplotlib package; NumPy diagnostics remain available.") from exc

    scores, _, _, explained = _shared_pca(flat)
    colors = {"known_gaussian": "#2878B5", "recovered_gaussian": "#E78824", "expert_inverse": "#4B9B69"}
    radii = {name: np.linalg.norm(array, axis=1) for name, array in flat.items()}
    all_radii = np.concatenate(list(radii.values()))
    low, high = float(all_radii.min()), float(all_radii.max())
    if low == high:
        low, high = max(0.0, low - 0.5), high + 0.5
    bins = np.linspace(low, high, 31)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    try:
        for name, array in flat.items():
            color = colors.get(name)
            axes[0].scatter(scores[name][:, 0], scores[name][:, 1], s=12, alpha=0.55, label=name, color=color)
            spectrum, rank, _ = _covariance_spectrum(array)
            # Include implicit zero modes when N < D so visible ranks refer to
            # the full feature space. symlog permits genuine zero eigenvalues.
            spectrum = np.pad(spectrum, (0, array.shape[1] - len(spectrum)))
            axes[1].plot(np.arange(1, len(spectrum) + 1), spectrum, label=f"{name} ({rank}/{array.shape[1]})", color=color)
            axes[2].hist(radii[name], bins=bins, density=True, histtype="step", linewidth=1.8, label=name, color=color)
        axes[0].set(title="PCA: one pooled center and basis", xlabel=f"PC1 ({100 * explained[0]:.1f}%)", ylabel=f"PC2 ({100 * explained[1]:.1f}%)")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[1].set(title="Covariance spectrum (legend: rank / D)", xlabel="Eigenvalue index", ylabel="Covariance eigenvalue")
        axes[1].set_yscale("symlog", linthresh=1e-8)
        axes[2].set(title="Radius in native coordinates", xlabel="L2 norm", ylabel="Density")
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.legend(fontsize=8)
        figure.suptitle("Source diagnostics — common coordinates, no per-group standardization", fontsize=12)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)
    finally:
        plt.close(figure)
    return str(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="NPZ containing the three named source groups")
    parser.add_argument("--outputdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-directions", type=int, default=128)
    parser.add_argument("--no-plot", action="store_true", help="Write JSON only")
    args = parser.parse_args(argv)
    try:
        with np.load(args.input, allow_pickle=False) as archive:
            missing = [key for key in SOURCE_KEYS if key not in archive]
            if missing:
                raise ValueError(f"NPZ is missing source keys: {', '.join(missing)}")
            arrays = {key: archive[key] for key in SOURCE_KEYS}
        report = source_diagnostics(arrays, seed=args.seed, n_directions=args.n_directions)
        args.outputdir.mkdir(parents=True, exist_ok=True)
        output = args.outputdir / "source_diagnostics.json"
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved diagnostics: {output}")
        if not args.no_plot:
            try:
                figure_path = plot_source_diagnostics(arrays, args.outputdir / "source_diagnostics.png")
                print(f"Saved figure: {figure_path}")
            except ImportError as exc:
                print(f"Figure skipped: {exc}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
